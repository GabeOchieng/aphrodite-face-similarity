import numpy as np
from base.BaseTrain import BaseTrain


class Trainer(BaseTrain):
    def __init__(self, sess, model, 
                 config, logger, data_loader):

        super(Trainer, self).__init__(
            sess, model, config, logger, data_loader)

        self.n_batch = config.trainer.batch_size

    def train_epoch(self, cur_epoch):
        losses = []
        accs = []

        for _ in range(self.config.trainer.num_iter_per_epoch):
            loss = self.train_step()
            losses.append(loss)
            print('loss: ' + str(loss))

        loss = np.mean(losses)
        
        cur_it = self.model.global_step_tensor.eval(self.sess)

        eval_loss = self.eval(cur_epoch, cur_it)

        summaries_dict = {
            'loss': loss,
            'eval_loss': eval_loss,
        }

        self.logger.summarize(
            cur_it, 
            summaries_dict=summaries_dict)

        self.model.save(self.sess)

        print('Average loss at epoch {0}: {1}'.\
            format(cur_epoch, loss))

        print('Average validation loss at epoch {0}: {1}'.\
            format(cur_epoch, eval_loss))

    def train_step(self):
        batch_l, batch_r, labels = \
            self.data_loader.get_siamese_batch(
                n=self.n_batch)

        feed_dict = {
            self.model.x_1: batch_l, 
            self.model.x_2: batch_r,
            self.model.y: labels,
            self.model.trainable: True
        }

        _, loss, step = self.sess.run(
            [
                self.model.train_step,
                self.model.loss,
                self.model.global_step_inc
            ],
            feed_dict=feed_dict)

        if (step + 1) % self.config.trainer.skip_step == 0:
            print('Loss at step {0}: {1}'.format(step, loss))

        return loss

    def eval(self, cur_epoch, cur_it):
        losses = []
        accs = []

        for i in range(self.config.trainer.num_iter_per_eval):
            loss = self.eval_step()
            losses.append(loss)

        loss = np.mean(losses)

        return loss

    def eval_step(self):
        batch_l, batch_r, labels = \
            self.data_loader.get_siamese_batch(
                n=1, trainable=False)
        
        feed_dict = {
            self.model.x_1: batch_l,
            self.model.x_2: batch_r, 
            self.model.y: labels,
            self.model.trainable: False
        }
        
        loss = self.sess.run(
            [
                self.model.loss
            ],
            feed_dict=feed_dict)

        return loss
