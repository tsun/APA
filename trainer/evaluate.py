import torch
import logging
import sklearn
import sklearn.metrics
from torch.autograd import Variable
from utils.utils import parse_address

def evaluate(model_instance, input_loader, domain):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    with torch.no_grad():
        for i in range(num_iter):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if model_instance.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            probabilities = model_instance.predict(inputs, domain)

            probabilities = probabilities.data.float()
            labels = labels.data.float()
            if first_test:
                all_probs = probabilities
                all_labels = labels
                first_test = False
            else:
                all_probs = torch.cat((all_probs, probabilities), 0)
                all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])

    # class-based average accuracy
    avg_acc = sklearn.metrics.balanced_accuracy_score(all_labels.cpu().numpy(),
                                                      torch.squeeze(predict).float().cpu().numpy())

    cm = sklearn.metrics.confusion_matrix(all_labels.cpu().numpy(),
                                          torch.squeeze(predict).float().cpu().numpy())
    accuracies = cm.diagonal() / cm.sum(1)
    precisions = cm.diagonal() / (cm.sum(0)+1e-6)
    predictions = cm.sum(0)

    f1s = 2/(1/(accuracies+1e-6) + 1/(precisions+1e-6))

    model_instance.set_train(ori_train_state)
    return {'accuracy': accuracy, 'per_class_accuracy': avg_acc, 'accuracies': accuracies, 'precisions':precisions, 'predictions':predictions, 'f1s':f1s}

def format_evaluate_result(eval_result):
    return 'Accuracy={}:Per-class accuracy={}:Accuracies={}:Precisions={}:Predictions={}:F1s={}'.format(
        eval_result['accuracy'].item(), eval_result['per_class_accuracy'], eval_result['accuracies'], eval_result['precisions'], eval_result['predictions'],eval_result['f1s'])

def evaluate_all(model_instance, val_source_loader, val_target_loader, test_loader, iter_num, args):
    if args.eval_source:
        eval_result = evaluate(model_instance, val_source_loader, 'source')
        if args.use_tensorboard:
            args.writer.add_scalar('source_accuracy', eval_result['accuracy'].item(), iter_num)
            args.writer.add_scalar('per_class_source_accuracy', eval_result['per_class_accuracy'], iter_num)
            args.writer.flush()
        print('\n')
        logging.info('Train iter={}:Source {}'.format(iter_num, format_evaluate_result(eval_result)))

    if args.eval_target and val_target_loader is not None:
        eval_result = evaluate(model_instance, val_target_loader, 'target')
        if args.use_tensorboard:
            args.writer.add_scalar('target_accuracy', eval_result['accuracy'].item(), iter_num)
            args.writer.add_scalar('per_class_target_accuracy', eval_result['per_class_accuracy'], iter_num)
            args.writer.flush()
        print('\n')
        logging.info('Train iter={}:Target {}'.format(iter_num, format_evaluate_result(eval_result)))

    if args.eval_test and test_loader is not None:
        if type(test_loader) is list or type(test_loader) is tuple:
            for i, t_extra_loader in enumerate(test_loader):
                eval_result = evaluate(model_instance, t_extra_loader, 'target')
                ext = parse_address(args.test_path[i])
                if args.use_tensorboard:
                    args.writer.add_scalar('test_accuracy_{}'.format(ext), eval_result['accuracy'].item(), iter_num)
                    args.writer.add_scalar('per_class_test_accuracy_{}'.format(ext), eval_result['per_class_accuracy'], iter_num)
                    args.writer.flush()
                print('\n')
                logging.info('Train iter={}:Test {} {}'.format(iter_num, ext, format_evaluate_result(eval_result)))
        else:
            eval_result = evaluate(model_instance, test_loader, 'target')
            if args.use_tensorboard:
                args.writer.add_scalar('test_accuracy', eval_result['accuracy'].item(), iter_num)
                args.writer.add_scalar('per_class_test_accuracy', eval_result['per_class_accuracy'], iter_num)
                args.writer.flush()
            print('\n')
            logging.info('Train iter={}:Test {}'.format(iter_num, format_evaluate_result(eval_result)))

def evaluate_from_dataloader(model_instance, input_loader, monte_carlo=False):
    with torch.no_grad():
        if monte_carlo is False:
            return evaluate_from_dataloader_basic(model_instance, input_loader)
        else:
            return evaluate_from_dataloader_monte_carlo(model_instance, input_loader)


def evaluate_from_dataloader_basic(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]

        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    predictions = torch.squeeze(predict).float()
    accuracy = torch.sum(predictions == all_labels).float() / float(all_labels.size()[0])

    # class-based average accuracy
    avg_acc = sklearn.metrics.balanced_accuracy_score(all_labels.cpu().numpy(), predictions.cpu().numpy())

    model_instance.set_train(ori_train_state)
    model_stats = {'accuracy': accuracy.item(),
                   'test_balanced_acc': avg_acc}
    return model_stats, all_probs

def evaluate_from_dataloader_monte_carlo(model_instance, input_loader):
    model_instance.set_train(True)
    all_probs = []
    for sample_i in range(model_instance.args.monte_carlo_sample_size):
        model_stats, probs_i = evaluate_from_dataloader_basic(model_instance, input_loader)
        all_probs.append(probs_i)
    probs = torch.stack(all_probs)
    probs_avg = probs.mean(dim=0)
    return model_stats, probs_avg

