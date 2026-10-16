import json
import os
import argparse
import re

def extract_label(response, choices):
    tmp_response = response.lower().strip()
    response = tmp_response.replace('*', '')
    if "a" == response:
        return "(a)"
    if "b" == response:
        return "(b)"
    if "c" == response:
        return "(c)"
    if "d" == response:
        return "(d)"
    if "unanswerable" == response:
        return "Unanswerable"
    def _extract_choice(response):
        for idx, choice in zip(["a", "b", "c", "d"], choices):
            pattern1 = rf'\b(answer|option|choice)\s+is\s+\(?{idx}\)?\b'
            pattern2 = rf'\b(answer|option|choice)\s*:\s*\(?{idx}\)?\b'
            pattern3 = rf'\b(answer|option|choice)\s+is\s+\(?{re.escape(choice)}\)?'
            pattern4 = rf'\b(answer|option|choice)\s*:\s*\(?{re.escape(choice)}\)?'
            pattern5 = rf"\b(answer|option|choice)\s+is\s+\$\\boxed{{\\text{{{re.escape(choice)}}}}}\$"
            for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5]:
                if re.search(pattern, response, re.I):
                    return f"({idx})"
        pattern1 = r'\b(answer|option|choice)\s+is\s+\(?unanswerable\)?\b'
        pattern2 = r'\b(answer|option|choice)\s*:\s*\(?unanswerable\)?\b'
        pattern3 = rf"\b(answer|option|choice)\s+is\s+\$\\boxed{{\\text{{unanswerable}}}}\$"
        for pattern in [pattern1, pattern2, pattern3]:
            if re.search(pattern, response, re.I):
                return "Unanswerable"
        return -1
    
    if "explanation" in response:
        first_part = response.split('explanation')[0]
    elif '\n' in response:
        first_part = response.split('\n')[0]
    else:
        first_part = None
    if first_part:
        extracted_label = _extract_choice(first_part)
        if extracted_label!=-1:
            return extracted_label
    if '\n' in response:
        last_part = response.split('\n')[-1]
        extracted_label = _extract_choice(last_part)        
        if extracted_label!=-1:
            return extracted_label
    
    extracted_label = _extract_choice(response)
    if extracted_label!=-1:
        return extracted_label

    def _extract_label(response):
        if "unanswerable" in response:
            return "Unanswerable"
        for idx, choice in zip(["(a)", "(b)", "(c)", "(d)"], choices):
            pattern = r'\b' + re.escape(choice)
            if re.search(pattern, response, re.I) is not None:
                return idx
        for idx, choice in zip(["(a)", "(b)", "(c)", "(d)"], choices):
            if idx in response:
                return idx
        return -1 
    if "explanation" in tmp_response:
        first_part = tmp_response.split('explanation')[0]
    elif '\n' in tmp_response:
        first_part = tmp_response.split('\n')[0]
    else:
        first_part = None
    if first_part:
        extracted_label = _extract_label(first_part)
        if extracted_label!=-1:
            return extracted_label
    if '\n' in tmp_response:
        last_part = tmp_response.split('\n')[-1]
        extracted_label = _extract_label(last_part)        
        if extracted_label!=-1:
            return extracted_label
    return _extract_label(tmp_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation.')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='UGeoQA') 
    args = parser.parse_args()
    assert args.dataset == 'UGeoQA'
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
            for _, line in enumerate(fr):
                total_num +=1
                d = json.loads(line)
                choices = json.load(open(d['filename'],'r'))["choices"]
                predicted_label = extract_label(d['response'], choices)
                gold_label = d['gold_label']
                # print(_+1, predicted_label, choices)
                if predicted_label == -1:
                    
                    failed += 1
                    predicted_label  = "(a)"
                    # print(_)
                    # exit()
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