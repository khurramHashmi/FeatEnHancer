import torch
import torch.nn as nn



class ScaleAwareFeatureAggregation(nn.Module):

    """
      ScaleAwareFeatureAggregation happens in five steps

      1 : Downgrade the size of two feature maps.
      2 : Converts the feature maps into query and key representaions.
      3 : Concats the downgraded features and divide it into blocks.
      4 : Performs Multi-Head Attention.
      5 : Aggregates and outputs the enhanced representation

    """


    def __init__(self, channels):
        super().__init__()

        self.mult_scale_heads = 8

        self.query_conv1 = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size= 7, stride=4)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
        self.key_conv = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size=3, stride=2)


    def forward(self, x, quarter_scale_x):
        """ 
        Args : 
            x:  a list of, feature map in (C, H, W) format.
            quarter_scale_x: a list of, quarter scale feature map in (C, H/4, W/4) format.

        Returns :
            aggregated_enhanced_representation : enhanced aggregated feature representations of
                                                 two feature maps from different scales.
                                                 a list of, enhanced features in (C, H, W)

        """

        orig_x = x

        # Key Query Generation
        x = self.query_conv1(x)
        x = self.query_conv2(x)
        quarter_scale_x = self.key_conv(quarter_scale_x)
        batch_size, C, roi_h, roi_w = x.size()
        
        x = x.view(batch_size, 1, C, roi_h, roi_w)
        quarter_scale_x = quarter_scale_x.view(batch_size, 1, C, roi_h, roi_w)

        x = torch.cat((x, quarter_scale_x), dim=1)
        batch_size, img_n, _, roi_h, roi_w = x.size()

        # Calculating the number of attention blocks
        x_embed = x
        c_embed = x_embed.size(2)

        # Performing multi-head attention
        # (img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        x_embed = x_embed.view(batch_size, img_n, self.mult_scale_heads, -1, roi_h,
                               roi_w)
        # (1, roi_n, num_attention_blocks, C / num_attention_blocks, H, W)
        target_x_embed = x_embed[:, [1]]
        # (batch_size, img_n, num_attention_blocks, 1, H, W)
        ada_weights = torch.sum(
            x_embed * target_x_embed, dim=3, keepdim=True) / (
                float(c_embed / self.mult_scale_heads)**0.5)
        # (batch_size, img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        ada_weights = ada_weights.expand(-1, -1, -1,
                                         int(c_embed / self.mult_scale_heads),
                                         -1, -1).contiguous()
        ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=1)

        # Aggregation and generation of enhanced representation
        x = (x * ada_weights).sum(dim=1)
        upsample = nn.UpsamplingBilinear2d((orig_x.size()[-2], orig_x.size()[-1]))
        aggregated_feature = upsample(x)
        aggregated_enhanced_representation = orig_x + aggregated_feature
        return aggregated_enhanced_representation

class FeatEnHancer(nn.Module):

    """
      Feature extraction for dark objects occurs in three stages
      1 : Extract low light features in three scales 
      2 : Scale aware aggregation of low light features 
      3 : Construction of enhanced representation of the input image


      Args : 
           in_channels : represent the number of channels in the input image (default is 3, considering RGB image)

    """

    def __init__(self, in_channels=3):
        super(FeatEnHancer, self).__init__()

        int_out_channels = 32
        out_channels = 24

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)

        # our convolution layers to transform the concatenated feature maps into the required feature shapes
        self.ue_conv8 = nn.Conv2d(out_channels*2, int_out_channels, 3, 1, 1, bias=True)
        self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)

        # Convolutions for downsampling orignal image into multiple scales
        self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
        self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)

        self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')


    def forward(self, x):

        """
        Args:
            x : a list, batched inputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                * image: Tensor, image in (C, H, W) format.
        
        Returns:
            enhanced_representation : a list, of enhanced representations
                                      Each representation is a Tensor, in (N, C, H, W) format.  
        """

        quarter_scale_x = self.quarter_conv(x)
        hexa_scale_x = self.hexa_conv(quarter_scale_x)
       
        # Extracting low light featurs at three different scales
        x1 = self.relu(self.e_conv1(x))
        quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
        hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))

        x2 = self.relu(self.e_conv2(x1))
        quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
        hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))

        x3 = self.relu(self.e_conv3(x2))
        quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
        hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))

        x4 = self.relu(self.e_conv4(x3))
        quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
        hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
        hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))

        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
        hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))

        x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
        quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
        hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))

        # Applying ScaleAwareFeatureAggregation between X7 and quarter_scale_x7
        x7 = self.feature_aggregation_block(x7, quarter_scale_x7)

        # Upsampling hexa scale to H x W for Skip connection
        x_upsample = nn.UpsamplingBilinear2d((x7.size()[-2], x7.size()[-1]))
        hexa_scale_x7 = x_upsample(hexa_scale_x7)
        x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))

        # Construction of Enhanced representation
        activated_enhanced_feature = torch.tanh(self.ue_conv9(x8))
        feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(activated_enhanced_feature, 3, dim=1)  
        x = x + feature1
        x = x + feature2
        x = x + feature3
        x = x + feature4
        x = x + feature5
        x = x + feature6
        x = x + feature7
        enhanced_representation = x + feature8
        return enhanced_representation
        
        

    def feature_aggregation_block(self, x, quarter_scale_x):

        # Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
        """
        Args:
            x:  a feature map in (C, H, W) format.
            quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.
        
        Returns:
            aggregated_enhanced_representation : Tensor, in (C, H, W) format.
        """
        aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
        return aggregated_enhanced_representation
