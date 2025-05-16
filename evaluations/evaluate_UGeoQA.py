import json
import os
import argparse

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
    parser.add_argument('--dataset', type=str, default='UgeoQA') 
    args = parser.parse_args()
    assert args.dataset == 'UgeoQA' 
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
            for line in fr:
                total_num +=1
                d = json.loads(line)
                choices = json.load(open(d['filename'],'r'))["choices"]
                predicted_label = extract_label(d['response'], choices)
                gold_label = d['gold_label']
                if predicted_label == -1:
                    failed += 1
                    predicted_label  = "(a)"
                    # print(total_num, gold_label, predicted_label)
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
        print(total_num)
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
        marco_precision = 0
        marco_recall = 0
        marco_f1_score = 0
        for c1 in ["(a)", "(b)", "(c)", "(d)", "Unanswerable"]:
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
            
            marco_precision += precision/5
            marco_recall += recall/5
            marco_f1_score += f1_score/5
            fw.write(f"-----class {c1}-----\n")
            fw.write(f"Precision for class {c1} {precision:.4f}\n")
            fw.write(f"Recall for class {c1} {recall:.4f}\n")
            fw.write(f"F1-score for class {c1} {f1_score:.4f}\n\n")
            
        # compute marco precision, recall, f1-score
        fw.write(f"Marco Precision {marco_precision:.4f}\n")
        fw.write(f"Marco Recall {marco_recall:.4f}\n")
        fw.write(f"Marco F1-score {marco_f1_score:.4f}\n\n")