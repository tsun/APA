from trainer.train import train_source_main
import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
import socket
hostName = socket.gethostname()

for i in [0]:
    print('random seed {}'.format(i))
    header = '''
            ++++++++++++++++++++++++++++++++++        
            {}
            ++++++++++++++++++++++++++++++++++
            @{}
            '''.format
    args = ['--base_model=Base'
                , '--gpu=0'
                , '--timestamp={}'.format(timestamp)
                , '--random_seed={}'.format(i)

                , '--base_net=ResNet50'
                , '--class_criterion=CrossEntropyLoss'

                , '--dataset=visda'
                , '--source_path=data/VisDA2017_train.txt'
                , '--test_path=[data/VisDA2017_valid.txt]'

                , '--train_source_sampler=ClassBalancedBatchSampler'
                , '--batch_size=64'

                , '--train_steps=20000'
                , '--save_interval=5000'
                , '--eval_interval=5000'

                , '--log_dir=log_base'
                , '--use_file_logger=True']

    train_source_main(args, header('\n\t'.join(args), hostName))