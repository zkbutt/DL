import torch

if __name__ == '__main__':
    '''excel:2维 -> sheet:3维 -> 多文件:4维 -> 多文件夹:5维 ...'''
    t1 = torch.arange(0, 12).reshape(-1)  # 12
    t2 = torch.arange(10, 14).reshape(-1)  # 4
    tc = torch.cat([t1, t2], dim=0)
    '''
    dim=0:运算后维度(16) 遍历0维
        取t1[0],t1[1] ... t2[0],t2[1]...
    '''
    t1 = torch.arange(0, 12).reshape((4, 3))
    t2 = torch.arange(10, 16).reshape((2, 3))
    tc = torch.cat([t1, t2], dim=0)
    '''
    dim=0:运算后维度(6,3) 遍历0维
        取t1[0,:],t1[1,:]...t1[3,:]
        取t2[0,:],t2[1,:]
    '''
    t1 = torch.arange(0, 12).reshape((3, 4))
    t2 = torch.arange(10, 16).reshape((3, 2))
    tc = torch.cat([t1, t2], dim=1)
    '''
    dim=1:运算后维度(3,6) 遍历1维
        t1[:,0] 取第一列 
        t1[:,1] 取第一列
        ...     ...
        t1[:,3] 取第一列
        t2[:,0] 取第一列
        t2[:,1] 取第二列
        取出来排好 结合运算后维度
    '''
    t1 = torch.arange(0, 12).reshape((2, 3, 2))
    t2 = torch.arange(10, 16).reshape((1, 3, 2))
    tc = torch.cat([t1, t2], dim=0)
    '''
     dim=1:运算后维度(3,3,2) 遍历0维
         t1[0,:,:] 取sheetA0 得(3, 2)
         t1[1,:,:] 取sheetA1 得(3, 2)
         t2[0,:,:] 取sheetB0 得(3, 2)
         取出来排好 结合运算后维度
     '''
    t1 = torch.arange(0, 12).reshape((2, 3, 2))
    t2 = torch.arange(10, 18).reshape((2, 2, 2))
    tc = torch.cat([t1, t2], dim=1)
    '''
    tensor([[[ 1.,  2.],
         [ 3.,  4.],
         [ 5.,  6.],
         [13., 14.],
         [15., 16.]],

        [[ 7.,  8.],
         [ 9., 10.],
         [11., 12.],
         [17., 18.],
         [19., 20.]]]) torch.Size([2, 5, 2])

     '''
    t1 = torch.arange(0, 12).reshape((2, 2, 3))
    t2 = torch.arange(10, 18).reshape((2, 2, 2))
    tc = torch.cat([t1, t2], dim=2)
    '''
    tensor([[[ 0,  1,  2, 10, 11],
         [ 3,  4,  5, 12, 13]],

        [[ 6,  7,  8, 14, 15],
         [ 9, 10, 11, 16, 17]]]) torch.Size([2, 2, 5])
    '''
    tc = torch.max(t1, dim=2) # 返回 values indices
    '''
    values=tensor([[ 2,  5],
        [ 8, 11]]),
    indices=tensor([[2, 2],
            [2, 2]]))
    '''
    tc = torch.argmax(t1, dim=2) # 返回 values indices
    '''
    tensor([[2, 2],
        [2, 2]])
    '''
    tc = torch.argsort(t1, dim=2) # 返回 values indices
    '''
    tensor([[[0, 1, 2],
         [0, 1, 2]],

        [[0, 1, 2],
         [0, 1, 2]]])
    '''
    print(tc)
    print(tc.values)
