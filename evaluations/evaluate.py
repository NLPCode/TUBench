import json
import os
import argparse
from nltk import word_tokenize 
def extract_label(response):
    # response = response.lower().strip().split('\n')[0]
    response = response.lower().strip()
    response = word_tokenize(response)
    if "yes" in response and "no" not in response and "unanswerable" not in response:
        return "Yes"
    elif "no" in response and "yes" not in response and "unanswerable" not in response:
        return "No"
    elif "unanswerable" in response or "it's impossible to say" in response:
        return "Unanswerable"
    elif "yes" in response:
        return "Yes"
    elif "no" in response:
        return "No"
    else:
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation.')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='UCR') 
    args = parser.parse_args()

    failed = 0
    total_num = 0
    answerable_num = 0
    total_acc = 0
    answerable_acc = 0
    answerable_2acc = 0
    unanswerable_2acc = 0
    two_label_acc = 0
    total_m = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                }
    # only for UCR
    if args.dataset == "UCR":
        case_m1 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m2 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m3 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m4 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
    if args.dataset == "UVQA":
        case_m1 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m2 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m3 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m4 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }
        case_m5 = { "Yes":          {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "No":           {"Yes": 0, "No": 0, "Unanswerable": 0},
                    "Unanswerable": {"Yes": 0, "No": 0, "Unanswerable": 0}
                    }

    output_filename = f"{args.dataset}/{os.path.basename(args.filename)}"
    os.makedirs(f"{args.dataset}", exist_ok=True)
    output_filename = output_filename.replace('jsonl', 'txt')
    if args.dataset == "UVQA":
        strategy_type_list = []
        for i in range(0, 300):
            cur_dir= f'../datasets/UVQA/select_val2014_vqa/{i}'
            if not os.path.exists(cur_dir):
                continue
            qa_filename = f'{cur_dir}/vqa.txt'
            with open(qa_filename, 'r') as fr:
                for line in fr:
                    contents = line.strip().split('\t')
                    if contents[2] == 'Unanswerable':
                        strategy_type = int(contents[3])
                    else:
                        strategy_type = -1
                    strategy_type_list.append(strategy_type)
    with open(output_filename, 'w') as fw:
        with open(args.filename, 'r') as fr:
            for idx, line in enumerate(fr):
                total_num +=1
                d = json.loads(line)
                predicted_label = extract_label(d['response'])
                
                gold_label = d['gold_label']
                if predicted_label == -1:
                    failed += 1
                    predicted_label  = "Yes"
                    # print(total_num, gold_label, predicted_label)
                if predicted_label == gold_label:
                    total_acc +=1
                    
                if gold_label in ["Yes", "No"]:
                    answerable_num += 1
                    if predicted_label==gold_label:
                        answerable_acc +=1
                    if predicted_label in ["Yes", "No"]:
                        answerable_2acc += 1
                else:
                    if predicted_label == "Unanswerable":
                        unanswerable_2acc += 1
                if predicted_label in ["Yes", "No"] and gold_label in ["Yes", "No"]:
                    two_label_acc += 1
                elif predicted_label == "Unanswerable" and gold_label == "Unanswerable":
                    two_label_acc += 1
                else:
                    pass
                total_m[gold_label][predicted_label] +=1
                if args.dataset == "UCR":
                    if "u1" in d["filename"]:
                        case_m1[gold_label][predicted_label] +=1
                    elif "u2" in d["filename"]:
                        case_m2[gold_label][predicted_label] +=1
                    elif "u3" in d["filename"]:
                        case_m3[gold_label][predicted_label] +=1
                    else:
                        case_m4[gold_label][predicted_label] +=1
                        
                if args.dataset == "UVQA" and gold_label == 'Unanswerable':
                    strategy_type = strategy_type_list[idx]
                    if strategy_type==1:
                        case_m1[gold_label][predicted_label] +=1
                    elif strategy_type==2:
                        case_m2[gold_label][predicted_label] +=1
                    elif strategy_type==3:
                        case_m3[gold_label][predicted_label] +=1
                    elif strategy_type==4:
                        case_m4[gold_label][predicted_label] +=1
                    elif strategy_type==5:
                        case_m5[gold_label][predicted_label] +=1
                    else:
                        print(strategy_type)
                        print(idx, line)
                        raise ValueError('')
                # print(predicted_label, gold_label)
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
        fw.write("Gold\Predict, Yes,  No,  Unanswerable\n")
        fw.write(f'Yes          , {total_m["Yes"]["Yes"]}, {total_m["Yes"]["No"]}, {total_m["Yes"]["Unanswerable"]}\n')
        fw.write(f'No           , {total_m["No"]["Yes"]}, {total_m["No"]["No"]}, {total_m["No"]["Unanswerable"]}\n')
        fw.write(f'Unanswerable , {total_m["Unanswerable"]["Yes"]}, {total_m["Unanswerable"]["No"]}, {total_m["Unanswerable"]["Unanswerable"]}\n\n') 
        
        # compute precision, recall, f1-score for every class
        marco_precision = 0
        marco_recall = 0
        marco_f1_score = 0
        for c1 in ["Yes", "No", "Unanswerable"]:
            fp = 0
            fn = 0
            for c2 in ["Yes", "No", "Unanswerable"]:
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
            
            marco_precision += precision/3
            marco_recall += recall/3
            marco_f1_score += f1_score/3
            fw.write(f"-----class {c1}-----\n")
            fw.write(f"Precision for class {c1} {precision:.4f}\n")
            fw.write(f"Recall for class {c1} {recall:.4f}\n")
            fw.write(f"F1-score for class {c1} {f1_score:.4f}\n\n")
            
        # compute marco precision, recall, f1-score
        fw.write(f"Marco Precision {marco_precision:.4f}\n")
        fw.write(f"Marco Recall {marco_recall:.4f}\n")
        fw.write(f"Marco F1-score {marco_f1_score:.4f}\n\n")
        
        if args.dataset == "UCR":
            total_acc = case_m4["Yes"]["Yes"] + case_m4["No"]["No"] + case_m4["Unanswerable"]["Unanswerable"]
            total_acc /= sum(case_m4["Yes"].values()) + sum(case_m4["No"].values()) + sum(case_m4["Unanswerable"].values())
            
            answerable_acc = case_m4["Yes"]["Yes"] + case_m4["No"]["No"]
            answerable_acc /= sum(case_m4["Yes"].values()) + sum(case_m4["No"].values())
            
            two_label_acc = case_m4["Yes"]["Yes"] + case_m4["No"]["No"] + case_m4["Yes"]["No"] + case_m4["No"]["Yes"] + case_m4["Unanswerable"]["Unanswerable"]
            two_label_acc /= sum(case_m4["Yes"].values()) + sum(case_m4["No"].values()) + sum(case_m4["Unanswerable"].values())
            
            
            fw.write("Confusion matrix for the normal case.\n")
            fw.write("Gold\Predict, Yes,  No,  Unanswerable\n")
            fw.write(f'Yes          , {case_m4["Yes"]["Yes"]}, {case_m4["Yes"]["No"]}, {case_m4["Yes"]["Unanswerable"]}\n')
            fw.write(f'No           , {case_m4["No"]["Yes"]}, {case_m4["No"]["No"]}, {case_m4["No"]["Unanswerable"]}\n')
            fw.write(f'Unanswerable , {case_m4["Unanswerable"]["Yes"]}, {case_m4["Unanswerable"]["No"]}, {case_m4["Unanswerable"]["Unanswerable"]}\n\n') 
            fw.write(f"two label accuracy: {two_label_acc:.4f}\n")
            fw.write(f"total label accuracy: {total_acc:.4f}\n")
            fw.write(f"answerable accuracy: {answerable_acc:.4f}\n\n")
                 
            
            total_acc = case_m1["Yes"]["Yes"] + case_m1["No"]["No"] + case_m1["Unanswerable"]["Unanswerable"]
            total_acc /= sum(case_m1["Yes"].values()) + sum(case_m1["No"].values()) + sum(case_m1["Unanswerable"].values())
            
            answerable_acc = case_m1["Yes"]["Yes"] + case_m1["No"]["No"]
            answerable_acc /= sum(case_m1["Yes"].values()) + sum(case_m1["No"].values())
            
            two_label_acc = case_m1["Yes"]["Yes"] + case_m1["No"]["No"] + case_m1["Yes"]["No"] + case_m1["No"]["Yes"] + case_m1["Unanswerable"]["Unanswerable"]
            two_label_acc /= sum(case_m1["Yes"].values()) + sum(case_m1["No"].values()) + sum(case_m1["Unanswerable"].values())
            
            recall = case_m1["Unanswerable"]["Unanswerable"]/sum(case_m1["Unanswerable"].values())
            if (case_m1["Unanswerable"]["Unanswerable"]+case_m1["Yes"]["Unanswerable"]+case_m1["No"]["Unanswerable"]) == 0:
                precision = 0
            else:
                precision = case_m1["Unanswerable"]["Unanswerable"]/(case_m1["Unanswerable"]["Unanswerable"]+case_m1["Yes"]["Unanswerable"]+case_m1["No"]["Unanswerable"])
            if precision+recall == 0:
                f1_score = 0
            else:
                f1_score = 2*precision*recall/(precision+recall)
            
            fw.write("Confusion matrix for the u1 case.\n")
            fw.write("Gold\Predict, Yes,  No,  Unanswerable\n")
            fw.write(f'Yes          , {case_m1["Yes"]["Yes"]}, {case_m1["Yes"]["No"]}, {case_m1["Yes"]["Unanswerable"]}\n')
            fw.write(f'No           , {case_m1["No"]["Yes"]}, {case_m1["No"]["No"]}, {case_m1["No"]["Unanswerable"]}\n')
            fw.write(f'Unanswerable , {case_m1["Unanswerable"]["Yes"]}, {case_m1["Unanswerable"]["No"]}, {case_m1["Unanswerable"]["Unanswerable"]}\n\n') 
            fw.write(f"F1-score for class Unanswerable {f1_score:.4f}\n")
            fw.write(f"two label accuracy: {two_label_acc:.4f}\n")
            fw.write(f"total label accuracy: {total_acc:.4f}\n")
            fw.write(f"answerable accuracy: {answerable_acc:.4f}\n\n")
            
            
            
            total_acc = case_m2["Yes"]["Yes"] + case_m2["No"]["No"] + case_m2["Unanswerable"]["Unanswerable"]
            total_acc /= sum(case_m2["Yes"].values()) + sum(case_m2["No"].values()) + sum(case_m2["Unanswerable"].values())
            
            answerable_acc = case_m2["Yes"]["Yes"] + case_m2["No"]["No"]
            answerable_acc /= sum(case_m2["Yes"].values()) + sum(case_m2["No"].values())
            
            two_label_acc = case_m2["Yes"]["Yes"] + case_m2["No"]["No"] + case_m2["Yes"]["No"] + case_m2["No"]["Yes"] + case_m2["Unanswerable"]["Unanswerable"]
            two_label_acc /= sum(case_m2["Yes"].values()) + sum(case_m2["No"].values()) + sum(case_m2["Unanswerable"].values())
            
            recall = case_m2["Unanswerable"]["Unanswerable"]/sum(case_m2["Unanswerable"].values())
            if (case_m2["Unanswerable"]["Unanswerable"]+case_m2["Yes"]["Unanswerable"]+case_m2["No"]["Unanswerable"]) == 0:
                precision = 0
            else:
                precision = case_m2["Unanswerable"]["Unanswerable"]/(case_m2["Unanswerable"]["Unanswerable"]+case_m2["Yes"]["Unanswerable"]+case_m2["No"]["Unanswerable"])
            if precision+recall == 0:
                f1_score = 0
            else:
                f1_score = 2*precision*recall/(precision+recall)
                
            fw.write("Confusion matrix for the u2 case.\n")
            fw.write("Gold\Predict, Yes,  No,  Unanswerable\n")
            fw.write(f'Yes          , {case_m2["Yes"]["Yes"]}, {case_m2["Yes"]["No"]}, {case_m2["Yes"]["Unanswerable"]}\n')
            fw.write(f'No           , {case_m2["No"]["Yes"]}, {case_m2["No"]["No"]}, {case_m2["No"]["Unanswerable"]}\n')
            fw.write(f'Unanswerable , {case_m2["Unanswerable"]["Yes"]}, {case_m2["Unanswerable"]["No"]}, {case_m2["Unanswerable"]["Unanswerable"]}\n\n') 
            fw.write(f"F1-score for class Unanswerable {f1_score:.4f}\n")
            fw.write(f"two label accuracy: {two_label_acc:.4f}\n")
            fw.write(f"total label accuracy: {total_acc:.4f}\n")
            fw.write(f"answerable accuracy: {answerable_acc:.4f}\n\n")
            
            
            total_acc = case_m3["Yes"]["Yes"] + case_m3["No"]["No"] + case_m3["Unanswerable"]["Unanswerable"]
            total_acc /= sum(case_m3["Yes"].values()) + sum(case_m3["No"].values()) + sum(case_m3["Unanswerable"].values())
            
            answerable_acc = case_m3["Yes"]["Yes"] + case_m3["No"]["No"]
            answerable_acc /= sum(case_m3["Yes"].values()) + sum(case_m3["No"].values())
            
            two_label_acc = case_m3["Yes"]["Yes"] + case_m3["No"]["No"] + case_m3["Yes"]["No"] + case_m3["No"]["Yes"] + case_m3["Unanswerable"]["Unanswerable"]
            two_label_acc /= sum(case_m3["Yes"].values()) + sum(case_m3["No"].values()) + sum(case_m3["Unanswerable"].values())
            
            recall = case_m3["Unanswerable"]["Unanswerable"]/sum(case_m3["Unanswerable"].values())
            if (case_m3["Unanswerable"]["Unanswerable"]+case_m3["Yes"]["Unanswerable"]+case_m3["No"]["Unanswerable"]) == 0:
                precision = 0
            else:
                precision = case_m3["Unanswerable"]["Unanswerable"]/(case_m3["Unanswerable"]["Unanswerable"]+case_m3["Yes"]["Unanswerable"]+case_m3["No"]["Unanswerable"])
            if precision+recall == 0:
                f1_score = 0
            else:
                f1_score = 2*precision*recall/(precision+recall)
                
            fw.write("Confusion matrix for the u3 case.\n")
            fw.write("Gold\Predict, Yes,  No,  Unanswerable\n")
            fw.write(f'Yes          , {case_m3["Yes"]["Yes"]}, {case_m3["Yes"]["No"]}, {case_m3["Yes"]["Unanswerable"]}\n')
            fw.write(f'No           , {case_m3["No"]["Yes"]}, {case_m3["No"]["No"]}, {case_m3["No"]["Unanswerable"]}\n')
            fw.write(f'Unanswerable , {case_m3["Unanswerable"]["Yes"]}, {case_m3["Unanswerable"]["No"]}, {case_m3["Unanswerable"]["Unanswerable"]}\n\n') 
            fw.write(f"F1-score for class Unanswerable {f1_score:.4f}\n")
            fw.write(f"two label accuracy: {two_label_acc:.4f}\n")
            fw.write(f"total label accuracy: {total_acc:.4f}\n")
            fw.write(f"answerable accuracy: {answerable_acc:.4f}\n\n")
        

        if args.dataset == "UVQA":
            for s_idx, case_m in enumerate([case_m1, case_m2, case_m3, case_m4, case_m5]):
                total_acc = case_m["Yes"]["Yes"] + case_m["No"]["No"] + case_m["Unanswerable"]["Unanswerable"]
                total_acc /= sum(case_m["Yes"].values()) + sum(case_m["No"].values()) + sum(case_m["Unanswerable"].values())
                
                two_label_acc = case_m["Yes"]["Yes"] + case_m["No"]["No"] + case_m["Yes"]["No"] + case_m["No"]["Yes"] + case_m["Unanswerable"]["Unanswerable"]
                two_label_acc /= sum(case_m["Yes"].values()) + sum(case_m["No"].values()) + sum(case_m["Unanswerable"].values())
                
                
                recall = case_m["Unanswerable"]["Unanswerable"]/sum(case_m["Unanswerable"].values())
                
                if (case_m["Unanswerable"]["Unanswerable"]+case_m["Yes"]["Unanswerable"]+case_m["No"]["Unanswerable"]) == 0:
                    precision = 0
                else:
                    precision = case_m["Unanswerable"]["Unanswerable"]/(case_m["Unanswerable"]["Unanswerable"]+case_m["Yes"]["Unanswerable"]+case_m["No"]["Unanswerable"])
                if precision+recall == 0:
                    f1_score = 0
                else:
                    f1_score = 2*precision*recall/(precision+recall)
                
                fw.write(f"Confusion matrix for the Strategy {s_idx+1}.\n")
                fw.write("Gold\Predict, Yes,  No,  Unanswerable\n")
                fw.write(f'Yes          , {case_m["Yes"]["Yes"]}, {case_m["Yes"]["No"]}, {case_m["Yes"]["Unanswerable"]}\n')
                fw.write(f'No           , {case_m["No"]["Yes"]}, {case_m["No"]["No"]}, {case_m["No"]["Unanswerable"]}\n')
                fw.write(f'Unanswerable , {case_m["Unanswerable"]["Yes"]}, {case_m["Unanswerable"]["No"]}, {case_m["Unanswerable"]["Unanswerable"]}\n\n') 
                fw.write(f"F1-score for class Unanswerable {f1_score:.4f}\n")
                fw.write(f"two label accuracy: {two_label_acc:.4f}\n")
                fw.write(f"total label accuracy: {total_acc:.4f}\n\n")
