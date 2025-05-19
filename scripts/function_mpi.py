#### MPI environments ####
from mpi4py import MPI

#### MPI functions ####
def para_range(n1, n2, nprocs, irank):

    iwork1 = (n2 - n1) // nprocs
    iwork2 = (n2 - n1) % nprocs
    ista   = irank * iwork1 + n1 + min(irank, iwork2)
    iend   = ista + iwork1
    if iwork2 > irank:
        iend = iend + 1
    return int(ista),int(iend)

def definetype(field_names, field_dtypes):
    num = nrows
    dtypes = list(zip(field_names, field_dtypes)) # zip to connect 2 arrays gradually
    a   = np.zeros(num, dtype = dtypes)

    struct_size = a.nbytes // num # // the floor division // rounds the result down to nearest int

    print(struct_size)
    offsets = [ a.dtype.fields[field][1] for field in field_names]

    mpitype_dict = {np.int32:MPI.INT,np.float64:MPI.DOUBLE,np.float32:MPI.REAL}
    field_mpitypes = [mpitype_dict[dtype] for dtype in field_dtypes]

    structtype = MPI.Datatype.Create_struct([1]*len(field_names), offsets, field_mpitypes)
    structtype = structtype.Create_resized(0, struct_size)
    structtype.Commit()
    return structtype


