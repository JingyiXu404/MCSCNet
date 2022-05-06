import numpy as np
import h5py
import cv2

def mat2npy(i,filename,name_noise,name_label,show=False,mean=0):
    #noise
    mat_noise = h5py.File(filename+name_noise+'_'+str(i)+'.mat','r')
    mat_t_noise = np.transpose(mat_noise[name_noise])
    mat_noise_out = np.transpose(mat_t_noise,(3,0,1,2))
    out_noise=mat_noise_out.astype('float32')
    out_noise=out_noise[:1,:,:,:]
    np.save(filename+name_noise+'_'+str(i)+'.npy', out_noise)
    print('noise shape:',out_noise.shape)
    #label
    mat_label = h5py.File(filename+ name_label+'_'+str(i)+ '.mat', 'r')
    mat_t_label = np.transpose(mat_label[name_label+str(noise)])
    mat_label_out = np.transpose(mat_t_label, (3, 0, 1, 2))
    out_label = mat_label_out.astype('float32')
    out_label = out_label[:1, :, :, :]
    np.save(filename+ name_label+'_'+str(i)+ '.npy', out_label)
    print('label shape:',out_label.shape)
    #show
    if show:
        data(out_label,out_noise,mean)
def data(gt,noise,mean=0):
    for j in range(gt.shape[0]):
        b=noise[j]
        c=gt[j]

        b=cv2.cvtColor(b,cv2.COLOR_RGB2BGR)+mean
        c=cv2.cvtColor(c,cv2.COLOR_RGB2BGR)+mean
        cv2.imshow('b',b)
        cv2.imshow('c',c)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    noise=70
    dataset='BSD68'
    filename='xxx'

    for i in range(24):
        j=i+1
        name_label = dataset + '_test_label'
        name_noise = dataset + '_test_noise'+str(noise)
        mat2npy(j,filename,name_noise,name_label,show=False,mean=0)
