import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import argparse
import re

parser = argparse.ArgumentParser(description="IBF Doc2vec experiment to measure patch similarity")
parser.add_argument("--mode", type=str, help="The train mode, should be one of: {.ast, .src, .ident}", choices=[".ast", ".src", ".ident"], required=True)
parser.add_argument("--input", type=str, help="The path to the input folder (it should contain the following subfolders: java_programs, patched_java_programs)", default = "./input")
parser.add_argument("--output", type=str, help="The path to the output folder", default = "./output")
args = parser.parse_args()


train_mode = args.mode
input = args.input
output = args.output

if not os.path.exists(output):
     os.makedirs(output)

# Splits the input string by the given pattern
def splitter(s):
    global train_mode
    if train_mode == ".ident" or train_mode == ".src":
        tokens = []
        splitted = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', s)
        for newtoken in splitted:
            tokens.append(newtoken.lower())
        return tokens
    else:
        return s.split()


print("Reading original programs...")
bugs = []
programs = dict()
train_sentences = []
for filename in os.listdir(input + "/java_programs/"):
    if not filename.endswith(".java"):
        continue

    bug_name = filename.replace(".java", "").lower()
    bugs.append(bug_name)

    with open(input + "/java_programs/" + bug_name.upper() + train_mode, "r", encoding="utf-8") as file:
        string_file = file.read()
        programs[bug_name] = string_file
        train_sentences.append(string_file)


print("Reading patched programs...")
patched_programs = dict()
for bug in bugs:
    patched_programs[bug] = dict()

    if not os.path.exists(input + "/patched_java_programs/" + bug):
        continue

    for tool in os.listdir(input + "/patched_java_programs/" + bug):
        patched_programs[bug][tool] = []

        for patch in os.listdir(input + "/patched_java_programs/" + bug + "/" + tool):

            full_path = input + "/patched_java_programs/" + bug + "/" + tool + "/" + patch + "/" + bug.upper() + train_mode

            if not os.path.exists(full_path):
                continue

            with open(full_path, "r", encoding="utf-8") as file:
                string_file = file.read()
                patched_programs[bug][tool].append(string_file) # split?
                train_sentences.append(string_file)


train_corpus = [TaggedDocument(words=splitter(s), tags=[str(i)]) for i, s in enumerate(train_sentences)]

# metaparameter values of the Doc2Vec model
epochs = 20
vec_size = 100
window_size = 10
min_count = 1

model = Doc2Vec(vector_size=vec_size,  min_count=min_count, epochs=epochs, workers=8, window_size=window_size)
model.build_vocab(train_corpus, keep_raw_vocab=True)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

print("Counting similarities...")
with open(output + "/similarities_" + train_mode.replace(".", "") + ".csv", "w", encoding="utf-8") as results:
    for bug, program in programs.items():

        if bug not in patched_programs:
            continue

        for tool, programs in patched_programs[bug].items():
            index = 1
            for patched_program in programs:

                if not patched_program or not program:
                    continue

                tokenized_program = splitter(program)
                tokenized_patched_program = splitter(patched_program)

                sim = model.n_similarity(tokenized_program, tokenized_patched_program)

                results.write(bug + "," + tool.lower() + "," + str(index) + "," + str(sim) + "\n")

                index += 1

print("The similarity file was saved to " + output + "/similarities_" + train_mode.replace(".", "") + ".csv")