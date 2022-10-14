from glob import glob

def get_it(path):
    return int(path.split('/')[-3][3:])

def get_pair(path):
    home,away = path.split('.')[0].split('/')[-1].split('-')
    return (int(home),int(away))

def sort_by_it(gifs):
    dec = [(-get_it(g),get_pair(g),g) for g in gifs]
    dec.sort()
    out = [g for it,pair,g in dec]
    return out

def generate(path=None, folder_name=None, env_name=None, algo_name=None, run_name=None):
    if path is None:
        # get the latest run
        # the path is env_name/algo_name/path
        path = sorted(glob(f'{folder_name}/{env_name}/{algo_name}/{run_name}'), key=os.path.getmtime)[-1]
    
    with open(f'{path}/gif_view.html', 'w') as f:
        cur_it = -1
        cur_player = -1
        f.write(f'<h1>Run name: {path}</h1><br>')
        # gifs = glob(f'{path}/renders/*.gif')
        gifs = glob(f'{path}/*/renders/*.gif')
        # for gif in sorted(gifs, key=lambda x:-int(x.split('/')[-3][3:])):
        for gif in sort_by_it(gifs):
            it = int(gif.split('/')[-3][3:])
            if it != cur_it:
                cur_it = it
                f.write(f'<br><h2>Iteration: {it}</h2><br>')
            player = int(gif.split('/')[-1].split('-')[0])
            if cur_player != player:
                cur_player = player
                f.write('<br>')
            img_path = "/".join(gif.split("/")[-3:])
            f.write(f'<img src="{img_path}", title="{img_path}", border="1", loading="lazy"></img>')
    return os.path.abspath(path)

if __name__=='__main__':
    import argparse
    import webbrowser
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--folder_name', type=str, default='results')
    parser.add_argument('--env_name', type=str, default='*')
    parser.add_argument('--algo_name', type=str, default='*')
    parser.add_argument('--run_name', type=str, default='*')
    args = parser.parse_args()

    # assert args.path ^ (args.env_name | args.algo_name | args.run_name)
    # path = args.path
    # if path is None:
    #     path = f'{args.env_name}/{args.algo_name}/{args.run_name}'
    path = generate(args.path, args.folder_name, args.env_name, args.algo_name, args.run_name)
    print(path)
    # cwd = os.path.dirname(os.path.realpath(__file__))
    webbrowser.open_new_tab(f'{path}/gif_view.html')