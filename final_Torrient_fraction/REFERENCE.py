 convert numpy array to tenor
# a  = torch.tensor(actions)

# note : This copy saved into desktop

 LIST TO TENSOR CONVERTION
# a = [1, 2, 3]
# b = torch.FloatTensor(a)


 REMOVE REPEATED ITEMS IN TENSOR
# a = [1, 3, 2, 3]
# c = torch.unique(torch.tensor(a))
# print(c)
#out : tensor([1, 2, 3])


 FIND TOP X VALUES IN LIST
# score = [350, 914, 569, 223, 947, 284, 567, 333, 697, 245, 227, 785, 120, 794, 343, 773, 293, 995]
# sorted(zip(score), reverse=True)[:3]


 FIND INDEX VALUES OF SPECIFIC
# my_list = ['A', 'B', 'C', 'D', 'E', 'F']
# print("The index of element C is ", my_list.index('C'))
# my_list.index('C')


GET INDICES OF THE N VALUES OF A LIST
# a = [5,3,1,4,10]
# sorted(range(len(a)), key=lambda i: a[i])[-2:]

### ANOTHER METHOD
# sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:2]

PRINT MAX VALUES OF THE INDEX
# x = torch.arange(1., 6.)
# x
# tensor([ 1.,  2.,  3.,  4.,  5.])
# torch.topk(x, 3)
# torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))

HOW TO CALL SHELL SCRIPT FROM PYTHON CODE ?
# import subprocess
# subprocess.call(['sh', './test.sh']) 


Pytorch tensor get the index of the element with specific values?
# a = torch.Tensor([1,2,2,3,4,4,4,5])
# b = torch.Tensor([1,2,4])
# np.in1d(a.numpy(), b.numpy())


Open a particular image from a path:

#img  = Image.open(path)     
# On successful execution of this statement,
# an object of Image type is returned and stored in img variable)
   
try: 
    img  = Image.open(path) 
except IOError:
    pass
# Use the above statement within try block, as it can 
# raise an IOError if file cannot be found, 
# or image cannot be opened.

CHoose Images with indexes

# >>> indexes = [2, 4, 5]

# >>> main_list = [0, 1, 9, 3, 2, 6, 1, 9, 8]

# >>> [main_list[x] for x in indexes]
# [9, 2, 6]
# >>> 

ADDING N WITH ALL THE ELEMENT IN LIST 

# np.add([1, 2, 3], 1).tolist()