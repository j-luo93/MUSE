
# coding: utf-8

# In[94]:

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
from src.evaluation.word_translation import get_nn_avg_dist
import numpy as np
import torch
import sys
import codecs
sys.path.insert(0,'../EMNLP-NMT/')

import vocab

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


_PARSER = None
def get_parser():
    global _PARSER
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    # mapping
    parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
    parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
    # discriminator
    parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
    parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
    parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
    parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
    parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
    parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
    parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
    parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
    # training adversarial
    parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
    parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    
    parser.add_argument("--full_vocab", action='store_true', help="Use full vocab (no lowercasing)")

    _PARSER = parser


# In[186]:

def word_translation(evaluator, input_wordlist, full_vocab=True):
    # mapped word embeddings
    src_emb = evaluator.mapping(evaluator.src_emb.weight).data
    tgt_emb = evaluator.tgt_emb.weight.data

    results = {}
    for method in ['nn', 'csls_knn_10']:
        results[method] = get_word_translation_accuracy(
            evaluator.src_dico.lang, evaluator.src_dico.word2id, src_emb,
            evaluator.tgt_dico.lang, evaluator.tgt_dico.word2id, tgt_emb,
            method=method,
            input_wordlist=input_wordlist,
            full_vocab=full_vocab
#             dico_eval=input_word'../DATA/wordlist.de.dev_not_in'
        )
    return results, src_emb, tgt_emb

def write_wv_to_file(load_path, vocab_path, output_path, size):
    print(load_path)
    vc, ic = torch.load(load_path)

    voc = vocab.Vocab(vocab_path)

    with codecs.open(output_path, 'w', 'utf8') as fout:
        fout.write('%d %d\n' %(len(voc), size))
        for v, i in zip(vc, ic):
            v = v.cpu().numpy()
            i = i.cpu().item()
            fout.write(voc[i] + ' ')
            for vi in v :
                fout.write('%.4f ' %vi)
            fout.write('\n')


def load_dictionary(input_wordlist, word2id, full_vocab=True):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
#     assert os.path.isfile(path)

    wordlist = []
    not_found = 0
    
    import io
#     with io.open(path, 'r', encoding='utf-8') as f:
#         for _, line in enumerate(f):
# #             assert line == line.lower()
#             word = line.lower().rstrip()
#             if word in word2id:
#                 wordlist.append(word)
#             else:
#                 not_found += 1
    for word in input_wordlist:
        if not full_vocab:
            word = word.lower()
        if word in word2id:
            wordlist.append(word)
        else:
            not_found += 1

    #logger.info("Found %i words in the dictionary (%i unique). "
    #            "%i other pairs contained at least one unknown word "
    #            % (len(wordlist), len(set([x for x in wordlist])),
    #               not_found))

    # do not sort the dictionary by source word frequencies
#     wordlist = sorted(wordlist, key=lambda x: word2id[x])
    dico = torch.LongTensor(len(wordlist))
    for i, word in enumerate(wordlist):
        dico[i] = word2id[word]

    return dico


# In[109]:

def get_models(params):
    assert not params.cuda or torch.cuda.is_available()
    assert 0 <= params.dis_dropout < 1
    assert 0 <= params.dis_input_dropout < 1
    assert 0 <= params.dis_smooth < 0.5
    assert params.dis_lambda > 0 and params.dis_steps > 0
    assert 0 < params.lr_shrink <= 1
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

    # build model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
    trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
    trainer.reload_best()
    
    evaluator = Evaluator(trainer)
    return evaluator, trainer


# In[190]:

def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, input_wordlist, full_vocab=True):
#     if dico_eval == 'default':
#         path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
#     else:
#         path = dico_eval
#     dico = load_dictionary(path, word2id1)
    dico = load_dictionary(input_wordlist, word2id1, full_vocab=full_vocab)
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico.max() < emb1.size(0)
#     assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico]
        scores = query.mm(emb2.transpose(0, 1))

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = torch.cat(word_scores, 1)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico][:, None])
        scores.sub_(average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(10, 1, True)
#     for k in [1, 5, 10]:
#         top_k_matches = top_matches[:, :k]
#         _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
#         # allow for multiple possible translations
#         matching = {}
#         for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
#             matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
#         # evaluate precision@k
#         precision_at_k = 100 * np.mean(list(matching.values()))
#         logger.info("%i source words - %s - Precision at k = %i: %f" %
#                     (len(matching), method, k, precision_at_k))

    return top_matches


# In[90]:

def get_saved_models(arg_string):
    global _PARSER
    params = _PARSER.parse_args(arg_string.split(' '))
    assert not hasattr(params, 'src_mean') and not hasattr(params, 'tgt_mean')
    return params, get_models(params)


# In[91]:

def get_test_models(arg_string, old_params):
    global _PARSER
    params = _PARSER.parse_args(arg_string.split(' '))
    if old_params.src_mean is not None:
        params.src_mean = old_params.src_mean.cuda()
    if old_params.tgt_mean is not None:
        params.tgt_mean = old_params.tgt_mean.cuda()
    return params, get_models(params)


# In[236]:

def pipeline(src_suf, tgt_suf, arg_string, size, old_params=None, data_dir='13-es-100K', center=False):
    import os
#     arg_string = '--exp_id k0gf0f007v --src_lang es --tgt_lang en '\
#                  '--n_refinement 5 --emb_dim 500 --normalize_embeddings center --full_vocab'
    for suf in [src_suf, tgt_suf]:
        if suf is None:
            continue
        load_path = '../EMNLP-NMT/data/%s/torch_save.%s' %(data_dir, suf)
        vocab_path = '../EMNLP-NMT/data/%s/%s' %(data_dir, suf)
        output_path = 'wv.%s' %suf
        if not os.path.isfile(output_path):
            write_wv_to_file(load_path, vocab_path, output_path, size)
        
        if suf == src_suf:
            arg_string += ' --src_emb %s' %output_path
        else:
            arg_string += ' --tgt_emb %s' %output_path
    
    if old_params is not None and center:
        print('getting test')
        lazy_test = LazyObject(lambda: get_test_models(arg_string, old_params.compute()[0]))
        #params, (test_eval, test_trainer) = get_test_models(arg_string, old_params.compute()[0])
    else:
        lazy_test = LazyObject(lambda: get_saved_models(arg_string))
        #params, (test_eval, test_trainer) = get_saved_models(arg_string)
    
    import torch
    import vocab
    
    for i, suf in enumerate([src_suf, tgt_suf]):
    #for emb, suf in zip([src_emb, tgt_emb], [src_suf, tgt_suf]):
        if suf is None:
            continue
        save_path = '../EMNLP-NMT/data/%s/torch_save.MUSE.%s' %(data_dir, suf)
        if os.path.isfile(save_path):
            continue
        
        
        if center:
            save_path += '.center'
        vocab_path = '../EMNLP-NMT/data/%s/%s' %(data_dir, suf)
        voc = vocab.Vocab(vocab_path)
        test_eval = lazy_test.compute()[1][0]
        
        if i == 0:
            emb = test_eval.src_emb
            # mapped word embeddings
            emb = test_eval.mapping(emb.weight).data
    
        else:
            emb = lazy_test.compute()[1][0].tgt_emb
            emb = emb.weight.data
            
        vc = emb.cpu()
        eval_vocab = test_eval.tgt_dico if suf == tgt_suf else test_eval.src_dico
        ic = torch.from_numpy(np.asarray([voc[w] for w in [eval_vocab.id2word[i] for i in range(len(vc))]]))
        torch.save([vc, ic], save_path)
    if 'test_eval' in locals():
        return test_eval
    else:
        return None


class LazyObject(object):
    
    def __init__(self, func):
        self.func = func
        self.ret = None
    
    def compute(self):
        if self.ret is None:
            self.ret = self.func()
        return self.ret

if __name__ == '__main__':
    get_parser()

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--exp_dir', type=str, required=True)
    main_parser.add_argument('--src_emb_path', type=str, required=True) 
    main_parser.add_argument('--src_vocab_path', nargs='*', type=str)
    main_parser.add_argument('--tgt_emb_path', type=str, required=True) 
    main_parser.add_argument('--tgt_vocab_path', nargs='*', type=str)
    main_parser.add_argument('--src_lang', type=str, required=True) 
    main_parser.add_argument('--tgt_lang', type=str, required=True) 
    main_parser.add_argument('--data_dir', type=str, required=True) 
    main_parser.add_argument('--center', action='store_true')
    main_parser.add_argument('--emb_dim', type=int, required=True)
    main_parser.add_argument('--test_words', type=str, nargs='*')
    args = main_parser.parse_args()

    args.exp_id = os.path.basename(args.exp_dir.strip('/'))
    
    old_arg_string = '--exp_id {exp_id} --src_lang {src_lang} --tgt_lang {tgt_lang} --src_emb {src_emb_path} --tgt_emb {tgt_emb_path} --n_refinement 5 --emb_dim {emb_dim}'.format(**vars(args))
    if args.center:
        old_arg_string += ' --normalize_embeddings center'

    lazy_saved_func = LazyObject(lambda: get_saved_models(old_arg_string))
    #old_params, (saved_eval, saved_trainer) = get_saved_models(old_arg_string)


    all_path_src = '../EMNLP-NMT/data/%s/torch_save.MUSE.all.%s' %(args.data_dir, args.src_lang)
    all_path_tgt = '../EMNLP-NMT/data/%s/torch_save.MUSE.all.%s' %(args.data_dir, args.tgt_lang)
    
    if args.center:
        all_path_src += '.center'
        all_path_tgt += '.center'

    def get_nns(evaluator, input_wordlist, method='nn', full_vocab=True):
        if not input_wordlist or evaluator is None: 
            print 'nothing to test'
            return None
        x, src_emb, tgt_emb = word_translation(evaluator, input_wordlist, full_vocab=full_vocab)
        y = x[method][1].view(-1).cpu().numpy()
        s = x[method][0].view(-1).cpu().numpy().reshape(-1, 10)
        z = np.asarray(map(lambda x: evaluator.tgt_dico.id2word[x], y)).reshape(-1, 10)
        print zip(z, s)

    if not os.path.isfile(all_path_tgt): # no need to save src
        #src_emb = saved_eval.mapping(saved_eval.src_emb.weight).data
        old_params, (saved_eval, saved_trainer) = lazy_saved_func.compute()
        tgt_emb = saved_eval.tgt_emb.weight.data.cpu()
        #src_words = [y for _, y in sorted(saved_eval.src_dico.id2word.items())]
        tgt_words = [y for _, y in sorted(saved_eval.tgt_dico.id2word.items())]
        #torch.save([src_emb, src_words], all_path_src)
        torch.save([tgt_emb, tgt_words], all_path_tgt)

        get_nns(saved_eval, args.test_words)

    if args.src_vocab_path:
        assert len(args.src_vocab_path) >= len(args.tgt_vocab_path)

        for svp, tvp in zip(args.src_vocab_path, args.tgt_vocab_path):
            svn = os.path.basename(svp)
            tvn = os.path.basename(tvp)
            new_arg_string = '--exp_id {exp_id} --src_lang {src_lang} --tgt_lang {tgt_lang} --n_refinement 5 --emb_dim {emb_dim}'.format(**vars(args))
            new_arg_string += ' --src_emb %s --tgt_emb %s --full_vocab' %(svn, tvn)
            if args.center:
                new_arg_string += ' --normalize_embeddings center '
            test_eval = pipeline(svn, tvn, new_arg_string, args.emb_dim,
                                    old_params=lazy_saved_func, data_dir=args.data_dir, center=args.center)


            get_nns(test_eval, args.test_words)

        for svp in args.src_vocab_path[len(args.tgt_vocab_path):]:
            svn = os.path.basename(svp)

            new_arg_string = '--exp_id {exp_id} --src_lang {src_lang} --tgt_lang {tgt_lang} --n_refinement 5 --emb_dim {emb_dim}'.format(**vars(args))
            new_arg_string += ' --src_emb %s --full_vocab' %svn
            if args.center:
                new_arg_string += ' --normalize_embeddings center'
            test_eval = pipeline(svn, None, new_arg_string, args.emb_dim,
                                    old_params=lazy_saved_func, data_dir=args.data_dir, center=args.center)
