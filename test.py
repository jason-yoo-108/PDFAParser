import argparse

import distance
import pandas as pd

from evaluate.beam import beam_search
from evaluate.eval_noiser import *
from evaluate.score import *
from infcomp import NameParser
from utilities.config import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json', type=str, default='config/v4.2.json')
parser.add_argument('--true_posterior', help='whether to sample from p(z|x) or q(z|x)', nargs='?', default=True,
                    type=bool)
parser.add_argument('--num_particles', help='# of particles to use for SIS', nargs='?', default=25, type=int)
parser.add_argument('--num_samples', help='# samples', nargs='?', default=10, type=int)
parser.add_argument('--parse', help='only parse instead of denoising and parsing', nargs='?', default=False, type=bool)
parser.add_argument('--test_set', help='path of the test set', nargs='?', default='data/british.csv')
parser.add_argument('--session_name', help='name of the save file in results folder', nargs='?', default='session')
parser.add_argument('--beam_width', help='beam width to be used during beam search', nargs='?', default=0, type=int)


args = parser.parse_args()
config = load_json(args.config)
name_parser = NameParser(config['rnn_num_layers'], config['rnn_hidden_size'], config['rnn_hidden_size'], peak_prob=1-1e-4)
name_parser.load_checkpoint(filename=f"{config['session_name']}")
name_parser.test_mode()

"""
COMMENT: Component accuracies measure % of the time the model got a component right if the component is in the input string
"""


def infer(observed_name, use_beam=False):
    if use_beam:
        # return beam search parse result
        parse_results = beam_search(name_parser, name_parser.index_encode(observed_name), args.beam_width)[0]
        return parse_results, {}
    
    if args.true_posterior:
        traces = get_importance_traces(observed_name, name_parser, args.num_samples, args.num_particles)
    else:
        traces = get_guide_traces(observed_name, name_parser, args.num_samples)
    log_probs = torch.Tensor(list(map(lambda trace: trace.log_prob_sum().item(), traces))).to(DEVICE)
    max_prob_trace = traces[torch.argmax(log_probs, dim=-1).item()]
    parse_results = get_parse_result(max_prob_trace)
    full_results = get_full_result(max_prob_trace, name_parser)

    return parse_results, full_results


def get_summary(test_df, clean_data):
    counter = 0

    title_parse_correct_count = 0
    fn_parse_correct_count = 0
    mn_parse_correct_count = 0
    ln_parse_correct_count = 0
    suffix_parse_correct_count = 0
    title_full_correct_count = 0
    fn_full_correct_count = 0
    mn_full_correct_count = 0
    ln_full_correct_count = 0
    suffix_full_correct_count = 0

    title_parse_distances = []
    fn_parse_distances = []
    mn_parse_distances = []
    ln_parse_distances = []
    suffix_parse_distances = []
    title_full_distances = []
    fn_full_distances = []
    mn_full_distances = []
    ln_full_distances = []
    suffix_full_distances = []

    correct_names_parse = []
    correct_noised_names_parse = []
    correct_titles_parse = []
    correct_fns_parse = []
    correct_mns_parse = []
    correct_lns_parse = []
    correct_suffixes_parse = []
    correct_names_full = []
    correct_noised_names_full = []
    correct_titles_full = []
    correct_fns_full = []
    correct_mns_full = []
    correct_lns_full = []
    correct_suffixes_full = []

    incorrect_names_parse = []
    incorrect_noised_names_parse = []
    incorrect_titles_parse = []
    incorrect_fns_parse = []
    incorrect_mns_parse = []
    incorrect_lns_parse = []
    incorrect_suffixes_parse = []
    incorrect_names_full = []
    incorrect_noised_names_full = []
    incorrect_titles_full = []
    incorrect_fns_full = []
    incorrect_mns_full = []
    incorrect_lns_full = []
    incorrect_suffixes_full = []

    for i, row in test_df.iterrows():

        if clean_data:
            input_string = row['name']
            correct_title_parse = row['title']
            correct_fn_parse = row['first']
            correct_mn_parse = row['middle']
            correct_ln_parse = row['last']
            correct_suffix_parse = row['suffix']
        else:
            input_string = row['name_noised']
            # If we are testing noised data + parsing, the parsing should simply return just the noised components
            correct_title_parse = row['title_noised']
            correct_fn_parse = row['first_noised']
            correct_mn_parse = row['middle_noised']
            correct_ln_parse = row['last_noised']
            correct_suffix_parse = row['suffix_noised']

        correct_title_full = row['title']
        correct_fn_full = row['first']
        correct_mn_full = row['middle']
        correct_ln_full = row['last']
        correct_suffix_full = row['suffix']

        parse_results, full_results = infer(input_string, use_beam=args.beam_width>0)

        pred_fn_parse, pred_mn_parse, pred_ln_parse = parse_results['firstname'], parse_results['middlename'], parse_results['lastname']
        pred_title_parse, pred_suffix_parse = parse_results['title'], parse_results['suffix']
        pred_fn_full, pred_mn_full, pred_ln_full = full_results['firstname'], full_results['middlename'], full_results['lastname']
        pred_title_full, pred_suffix_full = full_results['title'], full_results['suffix']

        title_dist_parse = distance.levenshtein(pred_title_parse, correct_title_parse)
        fn_dist_parse = distance.levenshtein(pred_fn_parse, correct_fn_parse)
        mn_dist_parse = distance.levenshtein(pred_mn_parse, correct_mn_parse)
        ln_dist_parse = distance.levenshtein(pred_ln_parse, correct_ln_parse)
        suffix_dist_parse = distance.levenshtein(pred_suffix_parse, correct_suffix_parse)
        title_dist_full = distance.levenshtein(pred_title_full, correct_title_full)
        fn_dist_full = distance.levenshtein(pred_fn_full, correct_fn_full)
        mn_dist_full = distance.levenshtein(pred_mn_full, correct_mn_full)
        ln_dist_full = distance.levenshtein(pred_ln_full, correct_ln_full)
        suffix_dist_full = distance.levenshtein(pred_suffix_full, correct_suffix_full)

        if len(correct_title_full) > 0:
            title_parse_distances.append(title_dist_parse)
            title_full_distances.append(title_dist_full)
            if title_dist_parse == 0: title_parse_correct_count += 1
            if title_dist_full == 0: title_full_correct_count += 1
        if len(correct_fn_full) > 0:
            fn_parse_distances.append(fn_dist_parse)
            fn_full_distances.append(fn_dist_full)
            if fn_dist_parse == 0: fn_parse_correct_count += 1
            if fn_dist_full == 0: fn_full_correct_count += 1
        if len(correct_mn_full) > 0:
            mn_parse_distances.append(mn_dist_parse)
            mn_full_distances.append(mn_dist_full)
            if mn_dist_parse == 0: mn_parse_correct_count += 1
            if mn_dist_full == 0: mn_full_correct_count += 1
        if len(correct_ln_full) > 0:
            ln_parse_distances.append(ln_dist_parse)
            ln_full_distances.append(ln_dist_full)
            if ln_dist_parse == 0: ln_parse_correct_count += 1
            if ln_dist_full == 0: ln_full_correct_count += 1
        if len(correct_suffix_full) > 0:
            suffix_parse_distances.append(suffix_dist_parse)
            suffix_full_distances.append(suffix_dist_full)
            if suffix_dist_parse == 0: suffix_parse_correct_count += 1
            if suffix_dist_full == 0: suffix_full_correct_count += 1

        if title_dist_parse > 0 or fn_dist_parse > 0 or mn_dist_parse > 0 or ln_dist_parse > 0 or suffix_dist_parse > 0:
            incorrect_names_parse.append(row['name'])
            incorrect_noised_names_parse.append(input_string)
            incorrect_titles_parse.append(pred_title_parse)
            incorrect_fns_parse.append(pred_fn_parse)
            incorrect_mns_parse.append(pred_mn_parse)
            incorrect_lns_parse.append(pred_ln_parse)
            incorrect_suffixes_parse.append(pred_suffix_parse)
        else:
            correct_names_parse.append(row['name'])
            correct_noised_names_parse.append(input_string)
            correct_titles_parse.append(pred_title_parse)
            correct_fns_parse.append(pred_fn_parse)
            correct_mns_parse.append(pred_mn_parse)
            correct_lns_parse.append(pred_ln_parse)
            correct_suffixes_parse.append(pred_suffix_parse)

        if title_dist_full > 0 or fn_dist_full > 0 or mn_dist_full > 0 or ln_dist_full > 0 or suffix_dist_full > 0:
            incorrect_names_full.append(row['name'])
            incorrect_noised_names_full.append(input_string)
            incorrect_titles_full.append(pred_title_full)
            incorrect_fns_full.append(pred_fn_full)
            incorrect_mns_full.append(pred_mn_full)
            incorrect_lns_full.append(pred_ln_full)
            incorrect_suffixes_full.append(pred_suffix_full)
        else:
            correct_names_full.append(row['name'])
            correct_noised_names_full.append(input_string)
            correct_titles_full.append(pred_title_full)
            correct_fns_full.append(pred_fn_full)
            correct_mns_full.append(pred_mn_full)
            correct_lns_full.append(pred_ln_full)
            correct_suffixes_full.append(pred_suffix_full)
        
        counter += 1
        if counter % 1 == 0: print(f"Counter: {counter}")

    accuracy_parse = len(correct_names_parse) / (len(correct_names_parse)+len(incorrect_names_parse))
    accuracy_full = len(correct_names_full) / (len(correct_names_full)+len(incorrect_names_full))

    title_accuracy_parse = title_parse_correct_count / len(title_parse_distances) if len(title_parse_distances)>0 else float('nan')
    fn_accuracy_parse = fn_parse_correct_count / len(fn_parse_distances)
    mn_accuracy_parse = mn_parse_correct_count / len(mn_parse_distances) if len(mn_parse_distances)>0 else float('nan')
    ln_accuracy_parse = ln_parse_correct_count / len(ln_parse_distances)
    suffix_accuracy_parse = suffix_parse_correct_count / len(suffix_parse_distances) if len(suffix_parse_distances)>0 else float('nan')
    title_avg_dist_parse = sum(title_parse_distances) / len(title_parse_distances) if len(title_parse_distances)>0 else float('nan')
    fn_avg_dist_parse = sum(fn_parse_distances) / len(fn_parse_distances)
    mn_avg_dist_parse = sum(mn_parse_distances) / len(mn_parse_distances) if len(mn_parse_distances)>0 else float('nan')
    ln_avg_dist_parse = sum(ln_parse_distances) / len(ln_parse_distances)
    suffix_avg_dist_parse = sum(suffix_parse_distances) / len(suffix_parse_distances) if len(suffix_parse_distances)>0 else float('nan')

    title_accuracy_full = title_full_correct_count / len(title_full_distances) if len(title_full_distances)>0 else float('nan')
    fn_accuracy_full = fn_full_correct_count / len(fn_full_distances)
    mn_accuracy_full = mn_full_correct_count / len(mn_full_distances) if len(mn_full_distances)>0 else float('nan')
    ln_accuracy_full = ln_full_correct_count / len(ln_full_distances)
    suffix_accuracy_full = suffix_full_correct_count / len(suffix_full_distances) if len(suffix_full_distances)>0 else float('nan')
    title_avg_dist_full = sum(title_full_distances) / len(title_full_distances) if len(title_full_distances)>0 else float('nan')
    fn_avg_dist_full = sum(fn_full_distances) / len(fn_full_distances)
    mn_avg_dist_full = sum(mn_full_distances) / len(mn_full_distances) if len(mn_full_distances)>0 else float('nan')
    ln_avg_dist_full = sum(ln_full_distances) / len(ln_full_distances)
    suffix_avg_dist_full = sum(suffix_full_distances) / len(suffix_full_distances) if len(suffix_full_distances)>0 else float('nan')

    result = {
        'accuracy_parse': accuracy_parse,
        'accuracy_full': accuracy_full,
        'title_accuracy_parse': title_accuracy_parse,
        'fn_accuracy_parse': fn_accuracy_parse,
        'mn_accuracy_parse': mn_accuracy_parse,
        'ln_accuracy_parse': ln_accuracy_parse,
        'suffix_accuracy_parse': suffix_accuracy_parse,
        'title_accuracy_full': title_accuracy_full,
        'fn_accuracy_full': fn_accuracy_full,
        'mn_accuracy_full': mn_accuracy_full,
        'ln_accuracy_full': ln_accuracy_full,
        'suffix_accuracy_full': suffix_accuracy_full,
        'title_avg_dist_parse': title_avg_dist_parse,
        'fn_avg_dist_parse': fn_avg_dist_parse,
        'mn_avg_dist_parse': mn_avg_dist_parse,
        'ln_avg_dist_parse': ln_avg_dist_parse,
        'suffix_avg_dist_parse': suffix_avg_dist_parse,
        'title_avg_dist_full': title_avg_dist_full,
        'fn_avg_dist_full': fn_avg_dist_full,
        'mn_avg_dist_full': mn_avg_dist_full,
        'ln_avg_dist_full': ln_avg_dist_full,
        'suffix_avg_dist_full': suffix_avg_dist_full,
    }

    def save_entries_csv(filepath, original, noised, t, f, m, l, s):
        df = pd.DataFrame({'original': original, 'noised': noised, 'predicted first': f, 'predicted middle': m, 
                            'predicted last': l, 'predicted title': t, 'predicted suffix': s})
        df.to_csv(filepath, index=None)

    status = 'clean' if clean_data else 'noised'

    save_entries_csv(
        f"result/entries/{args.session_name}_{status}_parse_correct.csv", 
        correct_names_parse, correct_names_parse if clean_data else correct_noised_names_parse, 
        correct_titles_parse, correct_fns_parse,
        correct_mns_parse, correct_lns_parse, correct_suffixes_parse
    )
    save_entries_csv(
        f"result/entries/{args.session_name}_{status}_parse_incorrect.csv", 
        incorrect_names_parse, incorrect_names_parse if clean_data else incorrect_noised_names_parse, 
        incorrect_titles_parse, incorrect_fns_parse,
        incorrect_mns_parse, incorrect_lns_parse, incorrect_suffixes_parse
    )
    save_entries_csv(
        f"result/entries/{args.session_name}_{status}_full_correct.csv", 
        correct_names_full, correct_names_full if clean_data else correct_noised_names_full, 
        correct_titles_full, correct_fns_full,
        correct_mns_full, correct_lns_full, correct_suffixes_full
    )
    save_entries_csv(
        f"result/entries/{args.session_name}_{status}_full_incorrect.csv", 
        incorrect_names_full, incorrect_names_full if clean_data else incorrect_noised_names_full,
        incorrect_titles_full, incorrect_fns_full,
        incorrect_mns_full, incorrect_lns_full, incorrect_suffixes_full
    )

    return result



test_data = pd.read_csv(args.test_set, keep_default_na=False)
clean_summary = get_summary(test_df=test_data, clean_data=True)
noised_summary = get_summary(test_df=test_data, clean_data=False)

description = list(clean_summary.keys())
summary = {'description': description,
           'clean': list(clean_summary.values()),
           'noised': list(noised_summary.values())}

pd.DataFrame(summary).to_csv(f"result/{args.session_name}.csv", index=None)
