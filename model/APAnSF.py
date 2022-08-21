import torch.nn as nn
import torch.nn.functional as F
import torch
from importlib import import_module

class APAnSF(object):
    def __init__(self, base_net='ResNet50', bottleneck_dim=1024, class_num=31, use_bottleneck=True, use_gpu=True, args=None):
        model_module = import_module('model.' + args.base_model)
        Model = getattr(model_module, args.base_model)
        self.c_net_base = Model(base_net, bottleneck_dim, class_num, use_bottleneck, use_gpu, args)
        self.c_net_feat = self.c_net_base.c_net_feat
        self.c_net_cls = self.c_net_base.c_net_cls

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num

        if self.use_gpu:
            self.c_net_feat = self.c_net_feat.cuda()
            self.c_net_cls = self.c_net_cls.cuda()

    def to_dicts(self):
        return [self.c_net_feat.state_dict(), self.c_net_cls.state_dict()]

    def from_dicts(self, dicts):
        self.c_net_feat.load_state_dict(dicts[0])
        self.c_net_cls.load_state_dict(dicts[1])

    def copy_from_source(self, source_model):
        self.c_net_feat.load_state_dict(source_model.c_net_feat.state_dict())
        self.c_net_cls.load_state_dict(source_model.c_net_cls.state_dict())


    def _l2_normalize(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def get_loss(self, inputs_source, inputs_target, labels_source, labels_target,
                            inputs_source_rand_aug, inputs_target_rand_aug, args=None):
        features_tgt = self.c_net_feat(inputs_target)
        outputs_tgt = self.c_net_cls(features_tgt)

        total_loss = 0.

        if args.self_training_loss_weight > 0:
            max_prob, pseudo_label = F.softmax(outputs_tgt).max(dim=-1)
            st_loss = (F.cross_entropy(outputs_tgt, pseudo_label,
                              reduction='none') * (max_prob >= args.self_training_conf).float().detach()).mean()
            total_loss += st_loss * args.self_training_loss_weight

        if args.vat_loss_weight > 0:
            features_tgt_norm = F.normalize(features_tgt)

            d = torch.rand(features_tgt.shape).sub(0.5).to(features_tgt.device)
            num_iters = args.VAT_iter
            xi = args.VAT_xi
            eps = args.VAT_eps
            pred = F.softmax(outputs_tgt).detach().clone()

            for i in range(num_iters):
                d = xi * self._l2_normalize(d)
                d.requires_grad_()
                #  _outputs = self.c_net_cls(features_tgt.detach() + d) -->
                _outputs = self.c_net_cls.classifier_layer(F.normalize(features_tgt_norm.detach() + d))
                logp_hat = F.log_softmax(_outputs/self.c_net_cls.temperature, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = self._l2_normalize(d.grad)
                self.c_net_cls.zero_grad()

            r_adv = d * eps
            # project perturbation to unit sphere
            r_adv = (F.normalize(features_tgt_norm + r_adv) - features_tgt_norm)
            act = features_tgt_norm + r_adv.detach().clone()
            _outputs = self.c_net_cls.classifier_layer(act)

            logp_hat = F.log_softmax(_outputs / self.c_net_cls.temperature, dim=1)
            adap_loss = F.kl_div(logp_hat, pred, reduction='batchmean')

            total_loss += adap_loss * args.vat_loss_weight


        if args.fixmatch_loss_weight > 0.0:
            target_rand_aug_size = len(inputs_target_rand_aug)
            if target_rand_aug_size > 0:
                features_target_rand_aug = self.c_net_feat(torch.cat(inputs_target_rand_aug))
                outputs_target_rand_aug = self.c_net_cls(features_target_rand_aug)
                outputs_target_rand_aug_list = outputs_target_rand_aug.chunk(target_rand_aug_size)

            aug_loss = 0.0
            for i in range(target_rand_aug_size):
                outputs_tgt_aug = outputs_target_rand_aug_list[i]
                max_prob, pred_u = torch.max(F.softmax(outputs_tgt), dim=-1)
                aug_loss += (F.cross_entropy(outputs_tgt_aug, pred_u,
                                      reduction='none') * max_prob.ge(args.fixmatch_loss_threshold).float().detach()).mean()

            total_loss += aug_loss * args.fixmatch_loss_weight

        self.iter_num += 1

        return total_loss

    def predict(self, inputs, output='prob'):
        features = self.c_net_feat(inputs)
        outputs = self.c_net_cls(features)
        if output == 'prob':
            softmax_outputs = F.softmax(outputs)
            return softmax_outputs
        elif output == 'score':
            return outputs
        elif output == 'score+feature':
            return outputs, features
        elif output == 'feature':
            return features
        else:
            raise NotImplementedError('Invalid output')

    def get_parameter_list(self):
        return self.c_net_feat.parameter_list + self.c_net_cls.parameter_list

    def set_train(self, mode):
        self.c_net_feat.train(mode)
        self.c_net_cls.train(mode)
        self.is_train = mode

