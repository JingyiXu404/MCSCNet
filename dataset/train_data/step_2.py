import numpy as np
import h5py
import cv2

def mat2npy(filename,name_noise,name_label,show=False,mean=0):
    #noise
    for i in range(1):
        name='_0'+str(i+1)
        name=''
        mat_noise = h5py.File(filename+name_noise+ name+ '.mat','r')
        mat_t_noise = np.transpose(mat_noise[name_noise+ name])
        mat_noise_out = np.transpose(mat_t_noise,(3,0,1,2))
        out_noise=mat_noise_out.astype('float32')

        #label
        mat_label = h5py.File(filename+ name_label+ name+ '.mat', 'r')
        mat_t_label = np.transpose(mat_label[name_label+ name])
        mat_label_out = np.transpose(mat_t_label, (3, 0, 1, 2))
        out_label = mat_label_out.astype('float32')
        if i == 0:
            out_noise_all = out_noise
            out_label_all = out_label
        else:
            out_noise_all = np.concatenate((out_noise_all, out_noise))
            out_label_all = np.concatenate((out_label_all, out_label))
        print(out_noise_all.shape)
        # show
        if show:
            data(out_label, out_noise, mean)
    np.save(filename + name_noise + '.npy', out_noise_all)
    print('noise shape:', out_noise_all.shape)
    np.save(filename+name_label+'.npy',out_label_all)
    print('label shape:',out_label_all.shape)
def data(gt,NI,mean=0):
    for j in range(2): #gt.shape[0]
        b=NI[j]
        c=gt[j]
        b=cv2.cvtColor(b,cv2.COLOR_RGB2BGR)+mean
        c=cv2.cvtColor(c,cv2.COLOR_RGB2BGR)+mean
        cv2.imshow('b',b)
        cv2.imshow('c',c)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    noise=75
    filename='xxx'
    name_label='data_label_'+str(noise)
    name_noise = 'data_noise_'+str(noise)
    mat2npy(filename,name_noise,name_label,show=False)
