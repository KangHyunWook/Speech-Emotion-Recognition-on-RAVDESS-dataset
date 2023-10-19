import os

def getPathList(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            pathList.extend(getPathList(full_path))
    return pathList

def getSubFileMap(all_files):
    file_map = {}

    for path in all_files:

        splits = path.split(os.path.sep)
        fn=splits[-1]
        if fn in ['readme.txt', 'label.mat']:
            continue
        current_subject = fn.split('_')[0]

        if current_subject not in file_map:
            file_map[current_subject] = []
        file_map[current_subject].append(path)
    return file_map
