import re

row = "hello 123 world 456"
reafter = re.sub(r'\d','',row)
print(reafter)