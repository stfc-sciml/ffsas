# process equations in USER-GUIDE.md for GitHub
from urllib.parse import quote

# read
with open('USER-GUIDE-latex-raw.md', 'r') as f:
    txt = f.read()

# special characters
txt = txt.replace('$$', '¥')
txt = txt.replace('$-th', '$ -th')
txt = txt.replace('$-based', '$ -based')
txt = txt.replace('$-norm', '$ -norm')
txt = txt.replace(' ', 'ø')
txt = txt.replace('\n', 'ñ')

# inline equations
split_inline = txt.split('$')
txt = ''
for i, eq in enumerate(split_inline):
    if i % 2 == 1:
        eq = quote(eq.replace('ø', ' '), safe='')
        txt += f'![eq](https://latex.codecogs.com/svg.image?\inline%20{eq})'
    else:
        txt += eq

# display equations
split_inline = txt.split('¥')
txt = ''
for i, eq in enumerate(split_inline):
    if i % 2 == 1:
        eq = eq.replace('ø', ' ')
        eq = eq.replace('ñ', '\n')
        eq = quote(eq)
        txt += '<p align="center">'
        txt += f'<img src="https://latex.codecogs.com/svg.image?{eq}">'
        txt += '</p>'
    else:
        txt += eq

# special characters
txt = txt.replace('ø', ' ')
txt = txt.replace('ñ', '\n')

# write
with open('../USER-GUIDE.md', 'w') as f:
    f.write(txt)
