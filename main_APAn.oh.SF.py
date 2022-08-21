from trainer.train import train_main
import time
timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
import socket
hostName = socket.gethostname()
import os
pid = os.getpid()

domains = ['Product', 'Art', 'Clipart', 'Real_World']

for src in domains:
    for tgt in domains:
        if src == tgt:
            continue

        header = '''
                ++++++++++++++++++++++++++++++++++        
                {}  
                ++++++++++++++++++++++++++++++++++
                @{}:{}
                '''.format
        args = ['--model=APAnSF'
                    , '--base_model=Base'
                    , '--gpu=0'
                    , '--timestamp={}'.format(timestamp)

                    , '--base_net=ResNet50'
                    , '--class_criterion=CrossEntropyLoss'

                    , '--dataset=Office-Home'
                    , '--source_path=data/{}.txt'.format(src)
                    , '--target_path=data/{}.txt'.format(tgt)

                    , '--train_target_sampler=ClassBalancedBatchSampler'
                    , '--batch_size=16'

                    , '--train_steps=10000'
                    , '--save_interval=10000'
                    , '--eval_interval=1000'
                    , '--yhat_update_freq=100'

                    , '--source_rand_aug_size=0'
                    , '--target_rand_aug_size=0'

                    , '--self_training_loss_weight=1.0'
                    , '--self_training_conf=0.75'

                    , '--vat_loss_weight=0.1'
                    # , '--fixmatch_loss_weight=0.1'
                    # , '--fixmatch_loss_threshold=0.0'

                    , '--VAT_eps=1.0'
                    , '--VAT_xi=1.0'

                    , '--log_dir=log/APA/'

                    , '--source_checkpoint=checkpoint_base/Base_Office-Home_{}_2022-00-00_00.00.00_0_5000.pth'.format(src)

                    , '--eval_source=False'
                    , '--use_file_logger=True']

        train_main(args, header('\n\t'.join(args), hostName, pid))