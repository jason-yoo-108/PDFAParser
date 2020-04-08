"""
1. Return just the parse based on format
2. Return probabilities associated with each trace in q
3. Return probabilities associated with each trace in p
Compute probability of each sample from q and take the max one
"""
import pyro

from const import *
from pdfa.symbol import *


def get_parse_result(sample_trace) -> dict:
    rv_names = sample_trace.stochastic_nodes
    title, first, middle, last, suffix = [], [], [], [], []
    for i in range(MAX_STRING_LEN):
        format_class = SYMBOL[sample_trace.nodes[f"{ADDRESS['format']}_{i}"]['value'].item()]
        if format_class == TITLE:
            title.append(i)
        elif format_class == FIRST:
            first.append(i)
        elif format_class == MIDDLE:
            middle.append(i)
        elif format_class == LAST:
            last.append(i)
        elif format_class == SUFFIX:
            suffix.append(i)

    prev_index = middle[0] - 1 if len(middle) > 0 else -1
    discontinuity_index = -1
    for i in middle:
        if i == prev_index + 1:
            prev_index = i
        else:
            discontinuity_index = i

    def index_to_component(name_index_tensor, indexes) -> str:
        component = ''
        for i in indexes:
            toadd = PRINTABLE[name_index_tensor[i]]
            if toadd != PAD and toadd != EOS: component += toadd
        return component

    name = sample_trace.nodes['_INPUT']['kwargs']['observations']['output']
    if discontinuity_index > 0:
        middle_component = index_to_component(name, middle[:discontinuity_index]) + ' ' + index_to_component(name,
                                                                                                             middle[
                                                                                                             discontinuity_index:])
    else:
        middle_component = index_to_component(name, middle)

    return {
        'title': index_to_component(name, title),
        'firstname': index_to_component(name, first),
        'middlename': middle_component,
        'lastname': index_to_component(name, last),
        'suffix': index_to_component(name, suffix),
    }


def get_full_result(sample_trace, name_parser) -> dict:
    rv_names = sample_trace.stochastic_nodes

    title = ''
    firstname = ''
    middlename = ''
    middlename1 = ''
    middlename2 = ''
    lastname = ''
    suffix = ''

    if ADDRESS['title'] in rv_names:
        title = TITLE_LIST[sample_trace.nodes[ADDRESS['title']]['value'].item()]
    if ADDRESS['suffix'] in rv_names:
        suffix = SUFFIX_LIST[sample_trace.nodes[ADDRESS['suffix']]['value'].item()]
    for i in range(MAX_OUTPUT_LEN):
        if f"{ADDRESS['firstname']}_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"{ADDRESS['firstname']}_{i}"]['value'].item()] != EOS:
                firstname += name_parser.output_chars[sample_trace.nodes[f"{ADDRESS['firstname']}_{i}"]['value'].item()]
        if f"{ADDRESS['middlename']}_0_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"{ADDRESS['middlename']}_0_{i}"]['value'].item()] != EOS:
                middlename1 += name_parser.output_chars[
                    sample_trace.nodes[f"{ADDRESS['middlename']}_0_{i}"]['value'].item()]
        if f"{ADDRESS['middlename']}_1_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"{ADDRESS['middlename']}_1_{i}"]['value'].item()] != EOS:
                middlename2 += name_parser.output_chars[
                    sample_trace.nodes[f"{ADDRESS['middlename']}_1_{i}"]['value'].item()]
        if f"{ADDRESS['lastname']}_{i}" in rv_names:
            if name_parser.output_chars[sample_trace.nodes[f"{ADDRESS['lastname']}_{i}"]['value'].item()] != EOS:
                lastname += name_parser.output_chars[sample_trace.nodes[f"{ADDRESS['lastname']}_{i}"]['value'].item()]

    if middlename1 != '' and middlename2 != '':
        middlename = middlename1 + ' ' + middlename2
    elif middlename1 != '':
        middlename = middlename1

    return {
        'title': title,
        'firstname': firstname,
        'middlename': middlename,
        'lastname': lastname,
        'suffix': suffix
    }


def get_importance_traces(name, name_parser, num_samples, num_particles) -> list:
    sample_traces = []
    csis = pyro.infer.CSIS(name_parser.model, name_parser.guide, pyro.optim.Adam({'lr': 0.001}),
                           num_inference_samples=num_particles)
    posterior = csis.run(observations={'output': name_parser.index_encode(name)})
    particle_weights = csis.get_normalized_weights()
    # print(f"Importance Weights: {particle_weights}")
    for _ in range(num_samples):
        sample_traces.append(posterior())
    return sample_traces


def get_guide_traces(name, name_parser, num_samples) -> list:
    sample_traces = []
    for _ in range(num_samples):
        trace = pyro.poutine.trace(name_parser.guide).get_trace(observations={'output': name_parser.index_encode(name)})
        sample_traces.append(trace)
    return sample_traces
