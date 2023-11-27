from train_model import Lab2Model

if __name__ == '__main__':
    
    # 待调试的超参数
    dropout = True
    ps = [0.0,0.1,0.3,0.5,0.7,0.9]
    normalize = True
    lrd = True
    residual_connection = True
    channels = [[3,64,128,256,512,256,128],
                [3,64,128,256,512,256],
                [3,64,128,256,512],
                [3,64,128,256,128],
                [3,64,128,256],
                [3,64,128]]
    
    # 最终选定的参数
    dropout = False
    p = 0.0
    normalize = True
    lrd = True
    residual_connection = True
    channel = [3,64,128,256,512]

    # 模型训练
    model = Lab2Model(batch_size=128, num_workers=4, seed=0)
    model.train(lr=0.001, epochs=80, device='cuda', wait=4, lrd=lrd, fig_name=f'final',
                p=p, channels=channel, residual_connection=residual_connection, 
                normalize=normalize, dropout=dropout)

    # 选择好超参数后，测试模型表现
    print('Test score:', model.test())