
import json

with open('data/individual_solution/Bit_Manipulation/2730.json', 'r') as f:
    data = json.load(f)

print(data)
with open('2730.py', 'w') as f:
    for item in data:
        print(item['question_id'])
        code = item['code_list'][0]
        for line_code in code.split('\n'):
            f.write(line_code + '\n')