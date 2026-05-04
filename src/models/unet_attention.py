import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # W_g: Processes the gating signal (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x: Processes the skip connection (from encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # psi: Calculates the final attention weights (0 to 1)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Add them together and apply ReLU
        psi = self.relu(g1 + x1)
        # Get attention mask (0 to 1)
        psi = self.psi(psi)
        # Multiply the mask by the original skip connection
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 1, hidden_features: list = None, use_skip_connections: bool = True, use_activation_after_upsampling: bool = False, use_attention: bool = True):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output channels
            hidden_features: List of feature maps at each level [64, 128, 256, 512]. Model adjusts accordingly
            use_skip_connections: Whether to use skip connections in the decoder
            use_attention: Whether to apply AttentionBlocks to the skip connections
        """
        super().__init__()
        
        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
        
        self.use_skip_connections = use_skip_connections
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.use_attention = use_attention and use_skip_connections # Attention requires skip connections

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        self.attention_blocks = nn.ModuleList() # list for attention blocks

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block: downsampling
        in_ch = input_channels
        for h_feature in hidden_features:
            # 4 blocks: n_channelsx64, 64x128, 128x256, 256x512
            self.encoder.append(self._double_conv_block(in_ch, h_feature))
            in_ch = h_feature
        
        # bottleneck block (bottom of U): 512x1024
        self.bottleneck_layer = self._double_conv_block(hidden_features[-1], hidden_features[-1] * 2)
        
        # decoder block: upsampling
        for h_feature in reversed(hidden_features):
            # 4 downsampling blocks: 1024x512, 512x256, 256x128, 128x64
            if self.use_activation_after_upsampling:
                self.decoder.append(
                    nn.Sequential(nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(h_feature),
                    nn.ReLU(inplace=True)))
            else:
                self.decoder.append(nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2))
            
            # Initialize Attention Block for this level
            if self.use_attention:
                # F_g (gating) and F_l (skip) both have 'h_feature' channels at this stage
                # F_int is conventionally half the channels
                self.attention_blocks.append(
                    AttentionBlock(F_g=h_feature, F_l=h_feature, F_int=h_feature // 2)
                )
            
            decoder_in_channels = h_feature * 2 if use_skip_connections else h_feature
            self.decoder.append(self._double_conv_block(decoder_in_channels, h_feature))

        # output layer
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
        skip_connections = []
        
        # Encoder: double conv block + maxpool
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.maxpool(x) # dowsnsample
        
        # Bottleneck
        x = self.bottleneck_layer(x)
        
        # reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # decoder part
        for i in range(len(self.decoder) // 2): #(0, 3)
            x = self.decoder[2 * i](x)  # upsample
            
            if self.use_skip_connections:
                skip_x = skip_connections[i]

                # Apply Attention
                if self.use_attention:
                    # x acts as the gating signal 'g', skip_x is the features 'x'
                    skip_x = self.attention_blocks[i](g=x, x=skip_x)

                x = torch.cat([skip_x, x], dim=1)
            
            
            x = self.decoder[2 * i + 1](x)  # double conv block
        
        # output layer
        x = self.out_conv(x)
        return x