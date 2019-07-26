import torch
from utils.data_processing import encode_tensor, decode_tensor
import unittest
test_tens = None


class TestTensorEncoding(unittest.TestCase):

    def setUp(self) -> None:
        self.test_tens = torch.tensor([[1., 2.], [2., 3.]])
        indices = torch.LongTensor([[0, 0, 1], [0, 1, 1]])
        values = torch.FloatTensor([2, 3, 4])
        size = [3, 3]
        self.test_sparse = torch.sparse_coo_tensor(indices, values, size)

    def tearDown(self) -> None:
        self.test_tens = None
        self.test_sparse = None

    def test_encode(self):
        reference = {'device': torch.device(type='cpu'), 'dtype': torch.float32, 'layout':
            torch.strided, 'grad': None, 'data': [[1.0, 2.0], [2.0, 3.0]]}
        self.assertEqual(reference, encode_tensor(self.test_tens))

    def test_sparse_encode(self):
        reference = {'device': torch.device(type='cpu'), 'dtype': torch.float32, 'layout':
                     torch.sparse_coo, 'grad': None, 'data': [[2.0, 3.0, 0.0],
                                                              [0.0, 4.0, 0.0],
                                                              [0.0, 0.0, 0.0]]}
        self.assertEqual(reference, encode_tensor(self.test_sparse))

    # def test_grad_encode(self): TODO re-implement with a gradient to store.
    #     test_tens.requires_grad_()
    #     print(encode_tensor(test_tens))

    def test_decode(self):

        self.assertEqual(self.test_tens.shape, decode_tensor(encode_tensor(self.test_tens)).shape)


if __name__ == '__main__':
    unittest.main()
