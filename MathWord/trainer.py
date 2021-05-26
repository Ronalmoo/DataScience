import os
from time import time

from utils.sentence_processing import *
from utils.logger import print_log, store_results
from utils.helper import save_checkpoint
from utils.evaluate import cal_score
from utils.confidence_estimation import *
from collections import OrderedDict

from models import Seq2SeqModel
from evaluate import run_validation

def build_model(config, voc1, voc2, device, logger, num_iters):
    '''
        Add Docstring
    '''
    model = Seq2SeqModel(config, voc1, voc2, device, logger, num_iters)
    model = model.to(device)

    return model


def train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, epoch_offset= 0, min_val_loss=float('inf'), max_val_bleu=0.0, max_val_acc = 0.0, min_train_loss=float('inf'), max_train_acc = 0.0, best_epoch = 0, writer= None):
    '''
        Add Docstring
    '''

    if config.histogram and config.save_writer and writer:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch_offset)
    
    estop_count=0
    
    for epoch in range(1, config.epochs + 1):
        od = OrderedDict()
        od['Epoch'] = epoch + epoch_offset
        print_log(logger, od)

        batch_num = 1
        train_loss_epoch = 0.0
        train_acc_epoch = 0.0
        train_acc_epoch_cnt = 0.0
        train_acc_epoch_tot = 0.0
        val_loss_epoch = 0.0

        start_time = time()
        total_batches = len(train_dataloader)

        for data in train_dataloader:
            ques = data['ques']

            sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
            sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
            sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device)

            nums = data['nums']
            ans = data['ans']

            model.train()

            loss = model.trainer(ques, sent1_var, sent2_var, input_len1, input_len2, config, device, logger)
            train_loss_epoch += loss

            if config.show_train_acc:
                model.eval()

                _, decoder_output, _ = model.greedy_decode(ques, sent1_var, sent2_var, input_len1, input_len2, validation=True)
                temp_acc_cnt, temp_acc_tot, _ = cal_score(decoder_output, nums, ans, data['eqn'])
                train_acc_epoch_cnt += temp_acc_cnt
                train_acc_epoch_tot += temp_acc_tot

            batch_num+=1
            print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

        train_loss_epoch = train_loss_epoch/len(train_dataloader)
        if config.show_train_acc:
            train_acc_epoch = train_acc_epoch_cnt/train_acc_epoch_tot
        else:
            train_acc_epoch = 0.0

        time_taken = (time() - start_time)/60.0

        if config.save_writer and writer:
            writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

        logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
        logger.debug('Starting Validation')

        val_bleu_epoch, val_loss_epoch, val_acc_epoch = run_validation(config=config, model=model, dataloader=val_dataloader, voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch)

        if train_loss_epoch < min_train_loss:
            min_train_loss = train_loss_epoch

        if train_acc_epoch > max_train_acc:
            max_train_acc = train_acc_epoch

        if val_bleu_epoch[0] > max_val_bleu:
            max_val_bleu = val_bleu_epoch[0]

        if val_loss_epoch < min_val_loss:
            min_val_loss = val_loss_epoch

        if val_acc_epoch > max_val_acc:
            max_val_acc = val_acc_epoch
            best_epoch = epoch + epoch_offset

            if config.separate_opt:
                state = {
                    'epoch' : epoch + epoch_offset,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'voc1': model.voc1,
                    'voc2': model.voc2,
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'emb_optimizer_state_dict': model.emb_optimizer.state_dict(),
                    'train_loss_epoch' : train_loss_epoch,
                    'min_train_loss' : min_train_loss,
                    'train_acc_epoch' : train_acc_epoch,
                    'max_train_acc' : max_train_acc,
                    'val_loss_epoch' : val_loss_epoch,
                    'min_val_loss' : min_val_loss,
                    'val_acc_epoch' : val_acc_epoch,
                    'max_val_acc' : max_val_acc,
                    'val_bleu_epoch': val_bleu_epoch[0],
                    'max_val_bleu': max_val_bleu
                }
            else:
                state = {
                    'epoch' : epoch + epoch_offset,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'voc1': model.voc1,
                    'voc2': model.voc2,
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'train_loss_epoch' : train_loss_epoch,
                    'min_train_loss' : min_train_loss,
                    'train_acc_epoch' : train_acc_epoch,
                    'max_train_acc' : max_train_acc,
                    'val_loss_epoch' : val_loss_epoch,
                    'min_val_loss' : min_val_loss,
                    'val_acc_epoch' : val_acc_epoch,
                    'max_val_acc' : max_val_acc,
                    'val_bleu_epoch': val_bleu_epoch[0],
                    'max_val_bleu': max_val_bleu
                }
            logger.debug('Validation Bleu: {}'.format(val_bleu_epoch[0]))

            if config.save_model:
                save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)
            estop_count = 0
        else:
            estop_count+=1

        if config.save_writer and writer:
            writer.add_scalar('loss/val_loss', val_loss_epoch, epoch + epoch_offset)
            writer.add_scalar('acc/val_score', val_bleu_epoch[0], epoch + epoch_offset)

        od = OrderedDict()
        od['Epoch'] = epoch + epoch_offset
        od['best_epoch'] = best_epoch
        od['train_loss_epoch'] = train_loss_epoch
        od['min_train_loss'] = min_train_loss
        od['val_loss_epoch']= val_loss_epoch
        od['min_val_loss']= min_val_loss
        od['train_acc_epoch'] = train_acc_epoch
        od['max_train_acc'] = max_train_acc
        od['val_acc_epoch'] = val_acc_epoch
        od['max_val_acc'] = max_val_acc
        od['val_bleu_epoch'] = val_bleu_epoch
        od['max_val_bleu'] = max_val_bleu
        print_log(logger, od)

        if config.histogram and config.save_writer and writer:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch + epoch_offset)

        if estop_count > config.early_stopping:
            logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
            break

    if config.save_writer:
        writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
        writer.close()

    logger.info('Training Completed for {} epochs'.format(config.epochs))

    if config.results:
        store_results(config, max_val_bleu, max_val_acc, min_val_loss, max_train_acc, min_train_loss, best_epoch)
        logger.info('Scores saved at {}'.format(config.result_path))

    return max_val_acc
