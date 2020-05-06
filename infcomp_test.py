import argparse

import distance
import pandas as pd

from evaluate.beam import beam_search
from evaluate.eval_noiser import *
from evaluate.score import *
from infcomp import NameParser
from utilities.config import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='filepath to config json', type=str, default='config/v4.1.json')
parser.add_argument('--true_posterior', help='whether to sample from p(z|x) or q(z|x)', nargs='?', default=False,
                    type=bool)
parser.add_argument('--num_particles', help='# of particles to use for SIS', nargs='?', default=10, type=int)
parser.add_argument('--num_samples', help='# samples', nargs='?', default=10, type=int)
parser.add_argument('--parse', help='only parse instead of denoising and parsing', nargs='?', default=False, type=bool)
parser.add_argument('--noised', help='whether to noise the observed name', nargs='?', default=False, type=bool)
parser.add_argument('--test_set', help='path of the test set', nargs='?', default='data/test.csv')
parser.add_argument('--session_name', help='name of the save file in results folder', nargs='?', default='session')
parser.add_argument('--beam_width', help='beam width to be used during beam search', nargs='?', default=0, type=int)

args = parser.parse_args()

config = load_json(args.config)

name_parser = NameParser(config['rnn_num_layers'], config['rnn_hidden_size'], config['rnn_hidden_size'], peak_prob=1-1e-4)
name_parser.load_checkpoint(filename=f"{config['session_name']}")
name_parser.test_mode()

fn_correct_count = 0
mn_correct_count = 0
ln_correct_count = 0
fn_distances = []
mn_distances = []
ln_distances = []

test_data = pd.read_csv(args.test_set, keep_default_na=False)


def parse_to_append(result):
    if type(result) == str:
        to_append = result
    elif len(result) == 0:
        to_append = ''
    else:
        to_append = result[0]
    return to_append


def infer(observed_name):
    if args.beam_width > 0:
        # return beam search parse result
        parse = beam_search(name_parser, name_parser.index_encode(observed_name), args.beam_width)[0]
        return parse['firstname'], parse['middlename'], parse['lastname']
    
    if args.true_posterior:
        traces = get_importance_traces(observed_name, name_parser, args.num_samples, args.num_particles)
    else:
        traces = get_guide_traces(observed_name, name_parser, args.num_samples)
    log_probs = torch.Tensor(list(map(lambda trace: trace.log_prob_sum().item(), traces))).to(DEVICE)
    max_prob_trace = traces[torch.argmax(log_probs, dim=-1).item()]
    if args.parse:
        parse = get_parse_result(max_prob_trace)
    else:
        parse = get_full_result(max_prob_trace, name_parser)
    firstname, middlename, lastname = parse['firstname'], parse['middlename'], parse['lastname']
    return firstname, middlename, lastname

incorrect_names = []
incorrect_noised_names = []
incorrect_fns = []
incorrect_mns = []
incorrect_lns = []

correct_names = []
noised_names = []
correct_fns = []
correct_mns = []
correct_lns = []

counter = 0

for i, j in test_data.iterrows():

    curr = j['name']
    if args.noised:
        allowed_noise = [c for c in string.ascii_letters + string.digits]
        curr = noise_name(curr, allowed_noise)
        if curr == j['name']: continue
    
    correct_fn = j['first']
    correct_mn = j['middle']
    correct_ln = j['last']

    firstname, middlename, lastname = infer(curr)

    fn_distance = distance.levenshtein(firstname, correct_fn)
    mn_distance = distance.levenshtein(middlename, correct_mn)
    ln_distance = distance.levenshtein(lastname, correct_ln)
    fn_distances.append(fn_distance)
    mn_distances.append(mn_distance)
    ln_distances.append(ln_distance)

    if fn_distance > 0 or mn_distance > 0 or ln_distance > 0:
        incorrect_names.append(j['name'])
        incorrect_noised_names.append(curr)
        incorrect_fns.append(firstname)
        incorrect_mns.append(middlename)
        incorrect_lns.append(lastname)
    else:
        correct_names.append(j['name'])
        noised_names.append(curr)
        correct_fns.append(firstname)
        correct_mns.append(middlename)
        correct_lns.append(lastname)
    
    if fn_distance == 0:
        fn_correct_count += 1
    if mn_distance == 0:
        mn_correct_count += 1
    if ln_distance == 0:
        ln_correct_count += 1
    
    counter += 1
    if counter % 1 == 0: print(f"Counter: {counter}")

fn_average_distance = sum(fn_distances) / len(fn_distances)
mn_average_distance = sum(mn_distances) / len(mn_distances)
ln_average_distance = sum(ln_distances) / len(ln_distances)
print("First name average number of letters wrong: %.3f" % fn_average_distance)
print("Middle name average number of letters wrong: %.3f" % mn_average_distance)
print("Last name average number of letters wrong: %.3f" % ln_average_distance)

fn_accuracy_rate = fn_correct_count / len(fn_distances)
mn_accuracy_rate = mn_correct_count / len(mn_distances)
ln_accuracy_rate = ln_correct_count / len(ln_distances)
print("First name accuracy: %.3f" % fn_accuracy_rate)
print("Middle name accuracy: %.3f" % mn_accuracy_rate)
print("Last name accuracy: %.3f" % ln_accuracy_rate)
print("Total name accuracy: %.3f" % (1-len(incorrect_names)/len(test_data)))


if args.noised:
    incorrect_df = pd.DataFrame({'original name': incorrect_names, 'noised name': incorrect_noised_names, 'predicted first': incorrect_fns, 'predicted middle': incorrect_mns, 'predicted last': incorrect_lns})
    incorrect_df.to_csv(f"result/{args.session_name}_incorrect.csv", index=None)

    correct_df = pd.DataFrame({'original name': correct_names, 'noised name': noised_names, 'predicted first': correct_fns, 'predicted middle': correct_mns, 'predicted last': correct_lns})
    correct_df.to_csv(f"result/{args.session_name}_correct.csv", index=None)
else:
    incorrect_df = pd.DataFrame({'original name': incorrect_names, 'predicted first': incorrect_fns, 'predicted middle': incorrect_mns, 'predicted last': incorrect_lns})
    incorrect_df.to_csv(f"result/{args.session_name}_incorrect.csv", index=None)

    correct_df = pd.DataFrame({'original name': correct_names, 'predicted first': correct_fns, 'predicted middle': correct_mns, 'predicted last': correct_lns})
    correct_df.to_csv(f"result/{args.session_name}_correct.csv", index=None)
