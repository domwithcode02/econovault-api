with open('main.py', 'r') as f:
    content = f.read()
    count = content.count('"""')
    print('Total triple quotes:', count)
    if count % 2 != 0:
        print('WARNING: Odd number of triple quotes found!')
    else:
        print('Triple quotes are balanced')