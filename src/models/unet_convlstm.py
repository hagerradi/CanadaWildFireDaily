import torch
import torch.nn as nn

from src.models.convlstm import ConvLSTM

class SpatiotemporalUNet(nn.Module):
    def __init__(self, input_channels: int = 20, num_classes: int = 1, hidden_features: list = None, use_skip_connections: bool = True):
        """_summary_

        Args:
            input_channels: the number of input channels Defaults to 20.
            num_classes: the number of classification classes. Defaults to 1.
            hidden_features: the number of hidden features. Defaults to None.
            use_skip_connections: toggle to activate skip connections. Defaults to True.
        """
        super().__init__()
        
        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
            
        self.use_skip_connections = use_skip_connections
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- ENCODER ---
        in_ch = input_channels
        for h_feature in hidden_features:
            self.encoder.append(self._double_conv_block(in_ch, h_feature))
            in_ch = h_feature
            
        # --- CONVLSTM BOTTLENECK ---
        self.bottleneck_lstm = ConvLSTM(
            input_dim=hidden_features[-1],
            hidden_dim=hidden_features[-1], 
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # --- DECODER ---
        for h_feature in reversed(hidden_features):
            self.decoder.append(nn.ConvTranspose2d(h_feature, h_feature, kernel_size=2, stride=2))
            # If using skip connections, input to double conv is 2 * h_feature
            decoder_in_channels = h_feature * 2 if use_skip_connections else h_feature
            self.decoder.append(self._double_conv_block(decoder_in_channels, h_feature // 2 if h_feature != hidden_features[0] else h_feature))
            
        # Fix the last layer channel mapping based on the reversed list logic
        # The above loop leaves the last output with `hidden_features[0]` channels (e.g., 64)
        self.out_conv = nn.Conv2d(hidden_features[0], num_classes, kernel_size=1)

    def _double_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        
        # Fold Time into Batch for efficient spatial encoding
        # Shape becomes: (Batch * Time, Channels, Height, Width)
        x = x.view(B * T, C, H, W)
        
        skip_connections = []
        
        # Encoder pass
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.maxpool(x)
            
        # Process Bottleneck temporally
        # x is of shape: (B * T, 512, H_bot, W_bot)
        _, C_bot, H_bot, W_bot = x.shape
        
        # Unfold back to sequence: (Batch, Time, Channels, H_bot, W_bot)
        x = x.view(B, T, C_bot, H_bot, W_bot)
        
        # Pass the entire 5D sequence through the ConvLSTM module
        # It returns: layer_output_list, last_state_list
        _, last_states = self.bottleneck_lstm(x)
        
        # last_states is a list of [h, c] for each layer. 
        # Grab the hidden state 'h' from the 1st (and only) layer.
        h = last_states[0][0]
            
        # The final hidden state 'h' contains the aggregated spatial-temporal 
        x = h
        
        # Decoder pass
        # We need the skip connections, but only from the LAST day in the sequence (Day T).
        # We don't want to skip-connect the past days into the future prediction.
        skip_connections = skip_connections[::-1]
        
        for i in range(len(self.decoder) // 2):
            x = self.decoder[2 * i](x)  # Upsample
            
            if self.use_skip_connections:
                skip_x = skip_connections[i]
                # skip_x has shape (B * T, C_skip, H_skip, W_skip). 
                # We extract only the last time step for each item in the batch.
                _, C_skip, H_skip, W_skip = skip_x.shape
                skip_x = skip_x.view(B, T, C_skip, H_skip, W_skip)
                last_day_skip = skip_x[:, -1, :, :, :]  # Grab Day T
                
                x = torch.cat([last_day_skip, x], dim=1)
                
            x = self.decoder[2 * i + 1](x)  # Double conv
            
        # Output layer
        out = self.out_conv(x)
        return out