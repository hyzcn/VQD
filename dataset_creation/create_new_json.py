import json
from spelling import correct_spellings

def vqd_correction_spelling(input_file, output_file):

    qs = json.load(open(input_file))

    def separate_word_n_special_char(word):
        spec_char = ""
        if ',' in word:
            word = word.replace(',', '')
            spec_char = ','
        elif '?' in word:
            word = word.replace('?', '')
            spec_char = '?'
        elif '.' in word:
            word = word.replace('.', '')
            spec_char = '.'
        elif '\'s' in word:
            word = word.replace('\'s', ' \'s')
            spec_char = '\'s'

        return word, spec_char

    for item in qs.keys():
        if item == 'questions_ids':
            for id, sent in qs[item].items():
                words = sent.split()
                for i in range(len(words)):
                    words[i], spec_char = separate_word_n_special_char(words[i])
                    if words[i] in correct_spellings:
                        correct_w = correct_spellings[words[i]]
                        if len(correct_w) == 1:
                            correct_word = correct_w[0] + spec_char
                        else:
                            correct_word = correct_w[0] + ' ' + correct_w[1] + spec_char
                        words[i] = correct_word
                    else:
                        words[i] = words[i] + spec_char
                sentence = " ".join(words)
                qs[item][id] = sentence

    with open(output_file, 'w') as fp:
        json.dump(qs, fp)
    print("DONE")


def vqd_correction_0_10_bbox(input_file, output_file):
    data = json.load(open(input_file))
    annt = data['annotations']
    max_bboxes = 10

    for image_id in annt.keys():
        stats = annt[image_id]
        qa_pair = stats['qa']
        indx = []
        qa_pair_list = []
        for i, qa in enumerate(qa_pair):
            if len(qa[0]) <= max_bboxes:
                indx.append(i)

        for i in indx:
            qa_pair_list.append(qa_pair[i])

        stats['qa'] = qa_pair_list

    data['annotations'] = annt

    with open(output_file, 'w') as fp:
        json.dump(data, fp)
    print("DONE!")


if __name__ == '__main__':

    vqd_correction_spelling(input_file='../dataset/vqd_final.json',
                            output_file='../dataset/vqd_correction.json')

    vqd_correction_0_10_bbox(input_file='../dataset/vqd_correction.json',
                             output_file='../dataset/vqd_correction_0_10_bbox.json')