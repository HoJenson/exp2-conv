from train_model import Lab2Model

if __name__ == '__main__':
    model = Lab2Model(batch_size=64, num_workers=8, seed=0)

    # 待调试的超参数
    dropout = True
    p = 0.2
    normalize = True
    lrd = True
    residual_connection = True
    channels = [3,64,128,256,512,256,128]

    # 最好还是使用 GPU, CPU 太慢了。。。
    model.train(lr=0.001, epochs=80, device='cuda', wait=4, lrd=lrd, fig_name='Final',
                p=p, channels=channels, residual_connection=residual_connection, 
                normalize=normalize, dropout=dropout)

    # 选择好超参数后，测试模型表现
    print('Test score:', model.test())