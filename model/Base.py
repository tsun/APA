import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F

class Base_cls(nn.Module):
    def __init__(self, dim=1024, class_num=31, use_hidden_layer=False, l2_normalize=False, temperature=1.0, use_dropout=False):
        super(Base_cls, self).__init__()
        if use_hidden_layer:
            self.classifier_layer_list = [nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(dim, class_num)] \
                if use_dropout else [nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, class_num)]
        else:
            self.classifier_layer_list = [nn.Linear(dim, class_num)]

        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)

        self.l2_normalize = l2_normalize
        self.temperature = temperature

        for layer in self.classifier_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)

        self.parameter_list = [{"params": self.classifier_layer.parameters(), "lr": 1}]

    def forward(self, features):
        if not self.l2_normalize:
            score = self.classifier_layer(features) / self.temperature
        else:
            if isinstance(self.classifier_layer, nn.Sequential):
                features = self.classifier_layer[:-1](features)
                features = F.normalize(features)
                score = self.classifier_layer[-1](features) / self.temperature
            else:
                features = F.normalize(features)
                score = self.classifier_layer(features) / self.temperature
        return score


class Base_feat(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, use_dropout=False):
        super(Base_feat, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.use_dropout = use_dropout

        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1}]

        if self.use_bottleneck:
            self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim),
                                          nn.BatchNorm1d(bottleneck_dim), nn.ReLU()]
            if self.use_dropout:
                self.bottleneck_layer_list.append(nn.Dropout(0.5))
            self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)

            self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
            self.bottleneck_layer[0].bias.data.fill_(0.1)

            self.parameter_list.append({"params":self.bottleneck_layer.parameters(), "lr":1})


    def freeze_param(self, freeze=True):
        for param in self.base_network.parameters():
            param.requires_grad = not freeze
        if self.use_bottleneck:
            for param in self.bottleneck_layer.parameters():
                param.requires_grad = not freeze

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        return features


class Base(object):
    def __init__(self, base_net='ResNet50', bottleneck_dim=1024, class_num=31, use_bottleneck=True, use_gpu=True, args=None):
        self.c_net_feat = Base_feat(base_net, use_bottleneck, bottleneck_dim, use_dropout=args.use_bottleneck_dropout)
        dim = bottleneck_dim if use_bottleneck else self.c_net_feat.base_network.output_num()
        self.c_net_cls = Base_cls(dim, class_num, use_hidden_layer=args.use_hidden_layer, use_dropout=args.use_dropout,
                        l2_normalize=args.l2_normalize, temperature=args.temperature)

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

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        features = self.c_net_feat(inputs)
        outputs = self.c_net_cls(features)
        return outputs

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

