from trainer.train import train_source_main
import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
import socket
hostName = socket.gethostname()

domains = ['sketch', 'clipart', 'painting', 'real']

for src in domains:
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


                    , '--dataset=DomainNet'
                    , '--source_path=data/{}_train_mini.txt'.format(src)
                    , '--test_path=[{}]'.format(','.join(['data/{}_test_mini.txt'.format(tst) for tst in domains if tst!=src]))

                    , '--train_source_sampler=ClassBalancedBatchSampler'
                    , '--batch_size=16'

                    , '--train_steps=10000'
                    , '--save_interval=5000'
                    , '--eval_interval=1000'

                    , '--log_dir=log_base'
                    , '--use_file_logger=True']

        train_source_main(args, header('\n\t'.join(args), hostName))