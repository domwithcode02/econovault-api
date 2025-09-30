with open('main.py', 'r') as f:
    lines = f.readlines()
    in_triple = False
    for i, line in enumerate(lines, 1):
        if '"""' in line:
            count = line.count('"""')
            if count % 2 != 0:
                in_triple = not in_triple
            print(f'Line {i}: {line.strip()} (Triple quotes: {count}, In triple: {in_triple})')
    if in_triple:
        print('ERROR: Unclosed triple quote found!')
    else:
        print('All triple quotes are properly closed')