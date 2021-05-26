import json
from typing import Dict
import subprocess
import os

def main(py_file):
    """
    py_file: python file
    """
    cmd = ['python3', py_file]

    fd_popen = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout
    data = fd_popen.read().strip()
    fd_popen.close()

    return float(data)


def score(answer: str, label: str):
    """
    answer: Path of user's prediction(submit)
    label: Path of gold answer 
    """

    with open(answer, 'r') as f:
        answer = json.load(f)
    
    with open(label, 'r') as f:
        label = json.load(f)
    
    answer_keys = answer.keys()
    total_score = []
    for ids in label:
        if ids in answer_keys:
            user_answer = answer[ids]["answer"]
            user_equation = answer[ids]["equation"]

            gold_answer = label[ids]["Answer"]
            gold_eqn = label[ids]["Equation"]
            gold_weight = label[ids]["weight"]

            correct_answer = 1 if float(user_answer) == float(gold_answer) else 0
  
            with open(f'{ids}.py', 'a') as f:
                    
                # f.write('def answer():\n\tresult = ' + user_equation + '\n\treturn result')
                # f.write('def answer():\n\tfor index in range(10, 14 ):\n\tif(11<index<13):\n\t\t\treturn (index)')
                # f.write(user_equation)
                f.write('print('+ user_equation + ')')
                f.write('\n')

            pred = main(f'{ids}.py')
            subprocess.call(['rm', f'{ids}.py'])

            # os.remove(f'script/{ids}.py')
            # import pdb; pdb.set_trace()

            # correct_equation = 1 if float(eval(user_equation)) == float(gold_answer) else 0
            correct_equation = 1 if pred == float(gold_answer) else 0

            score = gold_weight * (correct_answer + correct_equation)
            total_score.append(score)

    return sum(total_score)


if __name__ == "__main__":
    results = score("answer/answer.json", "data/mawps-asdiv-a_svamp/label.json")
    print(results)
    # answer = main()
    # print(answer)
    # def eval_equation(equation)
    #     "for index in range(10, 14 ):\n\t\tif(11<index<13):\n\n\nprint(index)"
    # with open('example.py', 'w') as f:
    #     f.write("for index in range(10, 14 ):\n\tif(11<index<13):\n\t\tprint(index)")
    
    # answer = "for index in range(10, 14 ):\n\tif(11<index<13):\n\t\tprint(index)"
    # result = exec("for index in range(10, 14 ):\n\tif(11<index<13):\n\t\tprint(index)")
    #     for index in range(10, 14):
    #         if(11<index<13):
    #             print(index)