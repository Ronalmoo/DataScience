import os
import pdb
from time import time
import json
import pandas as pd
# from tensorboardX import SummaryWriter

from utils.sentence_processing import *
from utils.logger import print_log
from utils.helper import bleu_scorer
from utils.evaluate import cal_score, stack_to_string, get_infix_eq
from utils.confidence_estimation import *
from collections import OrderedDict
from utils.evaluate import answer_to_json

def run_validation(config, model, dataloader, voc1, voc2, device, logger, epoch_num):
    batch_num = 1
    val_loss_epoch = 0.0
    val_bleu_epoch = 0.0
    val_acc_epoch = 0.0
    val_acc_epoch_cnt = 0.0
    val_acc_epoch_tot = 0.0

    model.eval()

    refs= []
    hyps= []

    if config.mode == 'test':
        questions, gen_eqns, act_eqns, scores = [], [], [], []

    display_n = config.batch_size

    with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
        f_out.write('---------------------------------------\n')
        f_out.write('Epoch: ' + str(epoch_num) + '\n')
        f_out.write('---------------------------------------\n')
    total_batches = len(dataloader)
    json_list = []

    for i, data in enumerate(dataloader):
        sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
        sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
        nums = data['nums']
        ans = data['ans']
        if config.grade_disp:
            grade = data['grade']
        if config.type_disp:
            type1 = data['type']
        if config.challenge_disp:
            type1 = data['type']
            var_type = data['var_type']
            annotator = data['annotator']
            alternate = data['alternate']

        ques = data['ques']

        sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

        val_loss, decoder_output, decoder_attn = model.greedy_decode(ques, sent1_var, sent2_var, input_len1, input_len2, validation=True)        

        temp_acc_cnt, temp_acc_eqn, temp_acc_tot, disp_corr, aligned = cal_score(decoder_output, nums, ans, data['eqn'])
        json_file = answer_to_json(aligned)
        json_list.append(json_file)
        val_acc_epoch_cnt += (temp_acc_cnt * 0.3 + temp_acc_eqn * 0.7)
        val_acc_epoch_tot += temp_acc_tot

        sent1s = idx_to_sents(voc1, sent1_var, no_eos= True)
        sent2s = idx_to_sents(voc2, sent2_var, no_eos= True)

        refs += [[' '.join(sent2s[i])] for i in range(sent2_var.size(1))]
        hyps += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]

        if config.mode == 'test':
            questions+= data['ques']
            gen_eqns += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]
            act_eqns += [' '.join(sent2s[i]) for i in range(sent2_var.size(1))]
            scores   += [cal_score([decoder_output[i]], [nums[i]], [ans[i]], [data['eqn'][i]])[0] for i in range(sent1_var.size(1))]

        with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
            f_out.write('Batch: ' + str(batch_num) + '\n')
            f_out.write('---------------------------------------\n')
            for i in range(len(sent1s[:display_n])):
                try:
                    f_out.write('Example: ' + str(i) + '\n')
                    if config.grade_disp:
                        f_out.write('Grade: ' + str(grade[i].item()) + '\n')
                    if config.type_disp:
                        f_out.write('Type: ' + str(type1[i]) + '\n')
                    f_out.write('Source: ' + stack_to_string(sent1s[i]) + '\n')
                    f_out.write('Target: ' + stack_to_string(sent2s[i]) + '\n')
                    f_out.write('Generated: ' + stack_to_string(decoder_output[i]) + '\n')
                    if config.challenge_disp:
                        f_out.write('Type: ' + str(type1[i]) + '\n')
                        f_out.write('Variation Type: ' + str(var_type[i]) + '\n')
                        f_out.write('Annotator: ' + str(annotator[i]) + '\n')
                        f_out.write('Alternate: ' + str(alternate[i].item()) + '\n')
                    if config.nums_disp:
                        src_nums = 0
                        tgt_nums = 0
                        pred_nums = 0
                        for k in range(len(sent1s[i])):
                            if sent1s[i][k][:6] == 'number':
                                src_nums += 1
                        for k in range(len(sent2s[i])):
                            if sent2s[i][k][:6] == 'number':
                                tgt_nums += 1
                        for k in range(len(decoder_output[i])):
                            if decoder_output[i][k][:6] == 'number':
                                pred_nums += 1
                        f_out.write('Numbers in question: ' + str(src_nums) + '\n')
                        f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
                        f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
                    f_out.write('Result: ' + str(disp_corr[i]) + '\n' + '\n')
                except:
                    logger.warning('Exception: Failed to generate')
                    pdb.set_trace()
                    break
            f_out.write('---------------------------------------\n')
            f_out.close()

        if batch_num % config.display_freq ==0:
            for i in range(len(sent1s[:display_n])):
                try:
                    od = OrderedDict()
                    logger.info('-------------------------------------')
                    od['Source'] = ' '.join(sent1s[i])

                    od['Target'] = ' '.join(sent2s[i])

                    od['Generated'] = ' '.join(decoder_output[i])
                    print_log(logger, od)
                    logger.info('-------------------------------------')
                except:
                    logger.warning('Exception: Failed to generate')
                    pdb.set_trace()
                    break

        val_loss_epoch += val_loss
        batch_num +=1
        print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

    m = dict(zip(range(1, len(json_list) + 1), json_list))
    
    
    with open('answer/answer.json', 'w') as f:
        json.dump(m, f, indent='\t')
    
    val_bleu_epoch = bleu_scorer(refs, hyps)
    if config.mode == 'test':
        # results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores]).transpose()
        # results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score']
        # csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
        # results_df.to_csv(csv_file_path, index = False)
        # return sum(scores)/len(scores)
        results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores]).transpose()
        results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score']
        csv_file_path = os.path.join(config.outputs_path, config.dataset+'.json')
        results_df.to_json(csv_file_path, orient='index')
        return sum(scores) / len(scores)
        

    val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot

    return val_bleu_epoch, val_loss_epoch/len(dataloader), val_acc_epoch

def estimate_confidence(config, model, dataloader, logger):
    
    questions    = []
    act_eqns     = []
    gen_eqns    = []
    scores        = []
    confs        = []
    batch_num = 0
    
    #Load training data (Will be useful for similarity based methods)
    train_df     = pd.read_csv(os.path.join('data',config.dataset,'train.csv'))
    train_ques    = train_df['Question'].values 
    
    total_batches = len(dataloader)
    logger.info("Beginning estimating confidence based on {} criteria".format(config.conf))
    start = time()
    for data in dataloader:
        ques, eqn, nums, ans = data['ques'], data['eqn'], data['nums'], data['ans']
        
        if config.conf == 'posterior':
            decoded_words, confidence = posterior_based_conf(ques, model)
        elif config.conf == 'similarity':
            decoded_words, confidence = similarity_based_conf(ques, train_ques, model, sim_criteria= config.sim_criteria)
        else:
            #TODO: Implement other methods
            raise ValueError("Other confidence methods not implemented yet. Use -conf posterior")
        
        if not config.adv:
            correct_or_not = [cal_score([decoded_words[i]], [nums[i]], [ans[i]])[0] for i in range(len(decoded_words))]
        else:
            correct_or_not = [-1 for i in range(len(decoded_words))]

        gen_eqn = [' '.join(words) for words in decoded_words]
        
        questions     += ques
        act_eqns    += eqn
        gen_eqns    += gen_eqn
        scores        += correct_or_not
        confs        += list(confidence)
        batch_num    += 1
        print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

    results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores, confs]).transpose()
    results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score', 'Confidence']
    if config.conf != 'similarity':
        csv_file_path = os.path.join('ConfidenceEstimates',config.dataset + '_' + config.run_name + '_' + config.conf + '.csv')
    else:
        csv_file_path = os.path.join('ConfidenceEstimates',config.dataset + '_' + config.run_name + '_' + config.conf + '_' + config.sim_criteria + '.csv')
    results_df.to_csv(csv_file_path)
    logger.info("Done in {} seconds".format(time() - start))

def get_hiddens(config, model, val_dataloader, voc1, voc2, device):
    batch_num =1
    
    model.eval()

    hiddens = []
    operands = []

    for data in val_dataloader:
        if len(data['ques']) == config.batch_size:
            sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
            sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
            nums = data['nums']
            ans = data['ans']

            ques = data['ques']

            sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

            hidden, decoder_output = model.obtain_hidden(config, ques, sent1_var, sent2_var, input_len1, input_len2)

            infix = get_infix_eq(decoder_output, nums)[0] # WORKS ONLY FOR BATCH SIZE 1
            words = infix.split()

            type_rep = []
            operand_types = []

            for w in range(len(words)):
                if words[w] == '/':
                    if words[w-1][0] == 'n':
                        operand_types.append(['dividend', words[w-1]])
                    if words[w+1][0] == 'n':
                        operand_types.append(['divisor', words[w+1]])
                elif words[w] == '-':
                    if words[w-1][0] == 'n':
                        operand_types.append(['minuend', words[w-1]])
                    if words[w+1][0] == 'n':
                        operand_types.append(['subtrahend', words[w+1]])

            for z in range(len(operand_types)):
                entity = operand_types[z][1]
                for y in range(len(hidden)):
                    if hidden[y][0] == entity:
                        type_rep.append([operand_types[z][0], hidden[y][1]])

            hiddens = hiddens + hidden
            operands = operands + type_rep

    return hiddens, operands
