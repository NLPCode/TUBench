import json
import os
import argparse
import sys
sys.path.append('../')

def extract_question_type_dict(question, choices):
    if "shortage or a surplus" in question:
        return 'shortage_surplus'
    elif "have enough" in question:
        return 'have_enough'
    elif "change" in question or "affect" in question or "further from zero" in question or "closest to zero" in question or "farther from sea level" in question:
        return "change"
    elif ("the most" in question or "least" in question or "the fewest" in question or "the highest" in question or "the youngest" in question) and len(choices)>2:
        return 'most_least'
    elif "cost less" in question or "less money" in question or "less aid" in question or "fewer" in question or "more" in question or "better" in question:
        return "more_less"
    elif "relation a function" in question:
        return "function"
    elif "function linear or nonlinear" in question:
        return "linear_nonlinear"
    elif "begin" in question or "end" in question or "wait" in question or "How soon" in question or "What time" in question or "When does" in question:
        return "time"
    else:
        # print(filename)
        print(question)
        # break
        # exit()
def extract_label(response, choices):
    response = response.lower().strip()
    if "(a)" in response or "a)" in response or "answer: a" in response or "option a" in response or "answer is a" in response:
        return "(a)"
    elif "(b)" in response or "b)" in response or "answer: b" in response or "option b" in response or "answer is b" in response:
        return "(b)"
    elif "(c)" in response or "c)" in response or "answer: c" in response or "option c" in response or "answer is c" in response:
        return "(c)"
    elif "(d)" in response or "d)" in response or "answer: d" in response or "option d" in response or "answer is d" in response:
        return "(d)"
    elif "unanswerable" in response:
        return "Unanswerable"
    elif len(response) == 1 and "a" in response:
        return "(a)"
    elif len(response) == 1 and "b" in response:
            return "(b)"
    elif len(response) == 1 and "c" in response:
        return "(c)"
    elif len(response) == 1 and "d" in response:
        return "(d)"
    elif len(response) == 2 and "a." in response:
        return "(a)"
    elif len(response) == 2 and "b." in response:
            return "(b)"
    elif len(response) == 2 and "c." in response:
        return "(c)"
    elif len(response) == 2 and "d." in response:
        return "(d)"
    else:
        for idx, choice in zip(["(a)", "(b)", "(c)", "(d)"], choices):
            if choice.lower() in response:
                return idx
        # print(choices)
        return -1 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation.')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='UTabMWP')
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=400)
    args = parser.parse_args()
    print(args)
    assert args.dataset == 'UTabMWP' 
    failed = 0
    total_num = 0
    answerable_num = 0
    total_acc = 0
    answerable_acc = 0
    answerable_2acc = 0
    unanswerable_2acc = 0
    two_label_acc = 0
    total_m = { "(a)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                "(b)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                "(c)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                "(d)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                "Unanswerable": {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0}
                }
    os.makedirs(f"{args.dataset}", exist_ok=True)
    output_filename = f"{args.dataset}/{os.path.basename(args.filename)}"
    output_filename = output_filename.replace('jsonl', 'txt')
    with open(output_filename, 'w') as fw:
        with open(args.filename, 'r') as fr:
            for idx, line in enumerate(fr):
                if idx<args.start_id:
                    continue
                if idx>=args.end_id:
                    break
                d = json.loads(line)
                # get choices
                choices = json.load(open(d['filename'],'r'))["choices"]
                predicted_label = extract_label(d['response'], choices)
                gold_label = d['gold_label']
                total_num +=1

                if predicted_label == -1:
                    failed += 1
                    # print(total_num, predicted_label, choices)
                    predicted_label  = "(a)"
                
                if predicted_label == gold_label:
                    total_acc +=1
                if gold_label in ["(a)", "(b)", "(c)", "(d)"]:
                    answerable_num += 1
                    if predicted_label == gold_label:
                        answerable_acc += 1
                    if predicted_label in ["(a)", "(b)", "(c)", "(d)"]:
                        answerable_2acc += 1
                else:
                    if predicted_label == "Unanswerable":
                        unanswerable_2acc += 1
                if predicted_label in ["(a)", "(b)", "(c)", "(d)"] and gold_label in ["(a)", "(b)", "(c)", "(d)"]:
                    two_label_acc += 1
                elif predicted_label == "Unanswerable"  and gold_label == "Unanswerable":
                    two_label_acc += 1
                else:
                    pass
                total_m[gold_label][predicted_label] +=1
        unanswerable_num = total_num - answerable_num
        fw.write(f"""{args.filename}, 
                 two label accuracy: {two_label_acc/total_num:.4f} 
                 total label accuracy: {total_acc/total_num:.4f} 
                 unanswerable 2accuracy: {unanswerable_2acc/unanswerable_num:.4f}
                 answerable 2accuracy: {0 if answerable_num==0 else answerable_2acc/answerable_num:.4f}
                 answerable accuracy: {0 if answerable_num==0 else answerable_acc/answerable_num:.4f} 
                 total questions: {total_num} 
                 answerable question: {answerable_num} 
                 failed extrated lables: {failed}\n\n""")
        fw.write("Confusion matrix for total.\n")
        fw.write("Gold\Predict, (a),  (b),  (c),  (d),  Unanswerable\n")
        fw.write(f'(a)          , {total_m["(a)"]["(a)"]}, {total_m["(a)"]["(b)"]}, {total_m["(a)"]["(c)"]}, {total_m["(a)"]["(d)"]}, {total_m["(a)"]["Unanswerable"]}\n')
        fw.write(f'(b)          , {total_m["(b)"]["(a)"]}, {total_m["(b)"]["(b)"]}, {total_m["(b)"]["(c)"]}, {total_m["(b)"]["(d)"]}, {total_m["(b)"]["Unanswerable"]}\n')
        fw.write(f'(c)          , {total_m["(c)"]["(a)"]}, {total_m["(c)"]["(b)"]}, {total_m["(c)"]["(c)"]}, {total_m["(c)"]["(d)"]}, {total_m["(c)"]["Unanswerable"]}\n')
        fw.write(f'(d)          , {total_m["(d)"]["(a)"]}, {total_m["(d)"]["(b)"]}, {total_m["(d)"]["(c)"]}, {total_m["(d)"]["(d)"]}, {total_m["(d)"]["Unanswerable"]}\n')
        fw.write(f'Unanswerable , {total_m["Unanswerable"]["(a)"]}, {total_m["Unanswerable"]["(b)"]}, {total_m["Unanswerable"]["(c)"]}, {total_m["Unanswerable"]["(d)"]}, {total_m["Unanswerable"]["Unanswerable"]}\n\n') 
        
        # compute precision, recall, f1-score for every class
        for c1 in ["Unanswerable"]:
            fp = 0
            fn = 0
            for c2 in ["(a)", "(b)", "(c)", "(d)", "Unanswerable"]:
                if c1!=c2:
                    fn += total_m[c1][c2]
                    fp += total_m[c2][c1]
                else:
                    tp = total_m[c1][c2]
            if tp+fp == 0:
                precision = 0
            else:
                precision = tp/(tp+fp)
            
            if tp+fn == 0:
                recall = 0
            else:
                recall = tp/(tp+fn)

            if precision+recall==0:
                f1_score = 0
            else:
                f1_score = 2*precision*recall/(precision+recall)
            
            fw.write(f"-----class {c1}-----\n")
            fw.write(f"Precision for class {c1} {precision:.4f}\n")
            fw.write(f"Recall for class {c1} {recall:.4f}\n")
            fw.write(f"F1-score for class {c1} {f1_score:.4f}\n\n")
            
    with open(output_filename, 'a') as fw:
        question_type_list= ['shortage_surplus', "have_enough", "most_least",
                 "more_less", "function", "linear_nonlinear", "change", "time"]

        for question_type in question_type_list:
            failed = 0
            total_num = 0
            answerabel_num = 0
            five_label_acc = 0
            four_label_acc = 0
            two_label_acc = 0
            total_m = { "(a)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                        "(b)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                        "(c)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                        "(d)":          {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0},
                        "Unanswerable": {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0, "Unanswerable": 0}
                        }
            with open(args.filename, 'r') as fr:
                for idx, line in enumerate(fr):
                    if idx<args.start_id:
                        continue
                    if idx>=args.end_id:
                        break
                    d = json.loads(line)
                    question = json.load(open(d['filename'],'r'))["question"]
                    # get choices
                    choices = json.load(open(d['filename'],'r'))["choices"]
                    # break
                    if question_type != extract_question_type_dict(question, choices):
                        continue
                    
                    predicted_label = extract_label(d['response'], choices)
                    gold_label = d['gold_label']
                    total_num +=1
                    if gold_label in ["(a)", "(b)", "(c)", "(d)"]:
                        answerabel_num += 1
                    
                        # if question_type == "have_enough" and predicted_label == "Unanswerable":
                        #     print(idx)
                        
                    if predicted_label == -1:
                        failed += 1
                        predicted_label  = "(a)"
                    if predicted_label == gold_label:
                        five_label_acc +=1
                    if predicted_label == gold_label  and gold_label in ["(a)", "(b)", "(c)", "(d)"]:
                        four_label_acc += 1
                    if predicted_label in ["(a)", "(b)", "(c)", "(d)"] and gold_label in ["(a)", "(b)", "(c)", "(d)"]:
                        two_label_acc += 1
                    elif predicted_label == "Unanswerable"  and gold_label == "Unanswerable":
                        two_label_acc += 1
                    else:
                        pass
                    total_m[gold_label][predicted_label] +=1
            if total_num == 0:
                continue
            fw.write(f"\nQuestion Type: {question_type}\n")
            fw.write(f"{args.filename}, two label (answerable/unanswerable) accuracy: {two_label_acc/total_num:.4f}\n")
            fw.write(f"four label accuracy: {four_label_acc/answerabel_num:.4f}\n")
            fw.write(f"five label accuracy: {five_label_acc/total_num:.4f}\n")
            fw.write(f"total labels: {total_num}, failed extrated lables: {failed}.\n\n")
            fw.write("Confusion matrix for total.\n")
            fw.write("Gold\Predict, (a),  (b),  (c),  (d),  Unanswerable\n")
            fw.write(f'(a)          , {total_m["(a)"]["(a)"]}, {total_m["(a)"]["(b)"]}, {total_m["(a)"]["(c)"]}, {total_m["(a)"]["(d)"]}, {total_m["(a)"]["Unanswerable"]}\n')
            fw.write(f'(b)          , {total_m["(b)"]["(a)"]}, {total_m["(b)"]["(b)"]}, {total_m["(b)"]["(c)"]}, {total_m["(b)"]["(d)"]}, {total_m["(b)"]["Unanswerable"]}\n')
            fw.write(f'(c)          , {total_m["(c)"]["(a)"]}, {total_m["(c)"]["(b)"]}, {total_m["(c)"]["(c)"]}, {total_m["(c)"]["(d)"]}, {total_m["(c)"]["Unanswerable"]}\n')
            fw.write(f'(d)          , {total_m["(d)"]["(a)"]}, {total_m["(d)"]["(b)"]}, {total_m["(d)"]["(c)"]}, {total_m["(d)"]["(d)"]}, {total_m["(d)"]["Unanswerable"]}\n')
            fw.write(f'Unanswerable , {total_m["Unanswerable"]["(a)"]}, {total_m["Unanswerable"]["(b)"]}, {total_m["Unanswerable"]["(c)"]}, {total_m["Unanswerable"]["(d)"]}, {total_m["Unanswerable"]["Unanswerable"]}\n\n') 
            
        
        

            