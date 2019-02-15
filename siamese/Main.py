import tensorflow as tf

from dataloader.SiameseDataLoader import SiameseDataLoader
from model.ConvNet import ConvNet
from trainer.Trainer import Trainer
from utils.Parser import process_config
from utils.Logger import Logger
from utils.Utils import create_dirs, get_args


def main():
    args = get_args()
    m_config = process_config(args.config)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # create_dirs([config.summary_dir, config.checkpoint_dir])
        data_loader = SiameseDataLoader(config=m_config)
        model = ConvNet(data_loader=data_loader, config=m_config)
        logger = Logger(sess=sess, config=m_config)

        trainer = Trainer(
            sess=sess,
            model=model,
            config=m_config,
            logger=logger,
            data_loader=data_loader)

        trainer.train()


if __name__ == '__main__':
    main()
