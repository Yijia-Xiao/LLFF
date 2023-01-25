import numpy as np
import os
import imageio


def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    data = poses
    # np.savetxt('W_001.txt', np.append(data[:, :-1, 0], [[0, 0, 0, 1]], axis=0), delimiter=' ')
    # np.savetxt('W_011.txt', np.append(data[:, :-1, 1], [[0, 0, 0, 1]], axis=0), delimiter=' ')
    num_samples = poses.shape[-1]
    # print(num_samples)
    os.system(f'mkdir -p {basedir}/pose')
    for i in range(num_samples):
        # np.savetxt(f'{basedir}/pose/W_{i}.txt', np.append(data[:, :-1, i], [[0, 0, 0, 1]], axis=0), delimiter=' ')
        np.savetxt(f'{basedir}/pose/W_{i}.txt', np.append(data[:, 1: , i], [[0, 0, 0, 1]], axis=0), delimiter=' ')
    return poses, bds, imgs


# from load import load_data
# import numpy as np
# data = load_data('./inputs/FPS/')[0]
# np.savetxt('W_001.txt', np.append(data[:, :-1, 0], [[0, 0, 0, 1]], axis=0), delimiter=' ')
# np.savetxt('W_011.txt', np.append(data[:, :-1, 1], [[0, 0, 0, 1]], axis=0), delimiter=' ')

# from load import load_data
# import numpy as np
# data = load_data('./inputs/FPS/')[0]
# np.savetxt('W_001.txt', np.append(data[:, 1:, 0], [[0, 0, 0, 1]], axis=0), delimiter=' ')
# np.savetxt('W_011.txt', np.append(data[:, 1:, 1], [[0, 0, 0, 1]], axis=0), delimiter=' ')

# np.append(data[:, :-1, 0], [[0, 0, 0, 1]], axis=0)
