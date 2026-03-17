import torch
# model_name = 'cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a'
model_name = 'cascade_mask_rcnn_swin_base_patch4_window7'
pretrained_weights  = torch.load(model_name+'.pth')

num_class =1
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.0.fc_reg.bias'].resize_(num_class*4)

pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.1.fc_reg.bias'].resize_(num_class*4)

pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_cls.weight'].resize_(num_class+1, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_cls.bias'].resize_(num_class+1)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['roi_head.bbox_head.2.fc_reg.bias'].resize_(num_class*4)

pretrained_weights['state_dict']['roi_head.mask_head.0.conv_logits.weight'].resize_(num_class, 256,1,1)
pretrained_weights['state_dict']['roi_head.mask_head.0.conv_logits.bias'].resize_(num_class)

pretrained_weights['state_dict']['roi_head.mask_head.1.conv_logits.weight'].resize_(num_class, 256,1,1)
pretrained_weights['state_dict']['roi_head.mask_head.1.conv_logits.bias'].resize_(num_class)

pretrained_weights['state_dict']['roi_head.mask_head.2.conv_logits.weight'].resize_(num_class, 256,1,1)
pretrained_weights['state_dict']['roi_head.mask_head.2.conv_logits.bias'].resize_(num_class)

torch.save(pretrained_weights, model_name+"_%d.pth"%num_class)