"""
GROWAT1D
library for GROundWATer modelling in 1D
2024-02-09
Georg Kaufmann
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse
import sys

#================================#
def readParameter1D(infile='GROWAT1D_parameter.in',path='work/',control=False):
    """
    ! read GROWAT1D parameter file
    ! input:
    !  (from file infile)
    ! output:
    !  xmin,xmax,nx         - min/max for x coordinate [m], discretisation
    !  whichtime            - flag for time units used
    !  time_start,time_end  - start/end point for time scale [s]
    !  time_step            - time step [s]
    !  time_scale           - scaling coefficient for user time scale
    ! use:
    !  xmin,xmax,nx,time_start,time_end,time_step,time_scale,whichtime = libGROWAT1D.readParameter1D()
    ! note:
    !  file structure given!
    !  uses readline(),variables come in as string,
    !  must be separated and converted ...
    """
    # read in data from file
    f = open(path+infile,'r')
    line = f.readline()
    xmin,xmax,nx = float(line.split()[0]),float(line.split()[1]),int(line.split()[2])
    line = f.readline()
    whichtime = line.split()[0]
    line = f.readline()
    time_start,time_end,time_step = float(line.split()[0]),float(line.split()[1]),float(line.split()[2])
    f.close()
    # convert times from user times to seconds, based on whichtime flag
    min2sec  = 60.               # seconds per minute
    hour2sec = 3600.             # seconds per hour
    day2sec  = 24.*3600.         # seconds per day
    year2sec = 365.25*day2sec    # seconds per year
    month2sec = year2sec/12.00   # seconds per month
    if (whichtime=='min'):   time_scale = min2sec
    if (whichtime=='hour'):  time_scale = hour2sec
    if (whichtime=='day'):   time_scale = day2sec
    if (whichtime=='month'): time_scale = month2sec
    if (whichtime=='year'):  time_scale = year2sec
    time_start *= time_scale
    time_end   *= time_scale
    time_step  *= time_scale
    # control output to screen
    if (control):
        print('== GROWAT1D ==')
        print('%20s %10.2f %10.2f' % ('xmin,xmax [m]:',xmin,xmax))
        print('%20s %10i' % ('nx:',nx))
        
        print('%20s %10.2f' % ('time_start['+whichtime+']:',time_start/time_scale))
        print('%20s %10.2f' % ('time_end['+whichtime+']:',time_end/time_scale))
        print('%20s %10.2f' % ('time_step['+whichtime+']:',time_step/time_scale))
        print('%20s %s' % ('whichtime:',whichtime))
    return xmin,xmax,nx,time_start,time_end,time_step,time_scale,whichtime


#================================#
def readBC1D(infile='GROWAT1D_bc.in',path='work/',control=False):
    """
    ! read GROWAT1D boundary conditions file
    ! input:
    !  (from file infile)
    ! output:
    !  dataBC      - array of boundary conditions
    ! use:
    !  dataBC = libGROWAT1D.readBC1D()
    ! note:
    !  uses np.loadtxt(), data read as (float) array
    !  first two lines are meta data and are mandatory!
    """
    # read in data from file
    dataBC = np.loadtxt(path+infile,skiprows=2)
    if (control):
        for i in range(dataBC.shape[0]):
            print('%20s %30s' % ('BC:',dataBC[i]))
    return dataBC


#================================#
def readMaterials1D(infile='GROWAT1D_materials.in',path='work/',control=False):
    """
    ! read GROWAT1D material areas file
    ! input:
    !  (from file infile)
    ! output:
    !  dataMAT      - array of boundary conditions
    ! use:
    !  dataMAT = libGROWAT1D.readMaterials1D()
    ! note:
    !  uses np.loadtxt(), data read as (float) array
    !  first two lines are meta data and are mandatory!
    """
    # read in data from file
    dataMAT = np.loadtxt(path+infile,skiprows=2,dtype='str')
    if (dataMAT.ndim == 1): dataMAT = np.array([dataMAT])
    if (control):
        for i in range(dataMAT.shape[0]):
            print('%20s %30s' % ('Materials:',dataMAT[i]))
    return dataMAT


#================================#
def createNodes1D(xmin,xmax,nx,control=False):
    """
    !-----------------------------------------------------------------------
    ! create nodes in 1D domain
    ! first, a rectangular box with the given limits
    ! input:
    !  xmin,xmax         - min/max for x
    !  nx,ny             - number of nodes in x,y direction
    ! output:
    !  X                 - x-coordinate [m] of node i
    !  dx,dy             - spatial discretisation [m]
    ! use:
    !  X,dx = libGROWAT1D.createNodes1D(xmin,xmax,nx)
    ! notes:
    !  ij,ijk            - node counter in x/y and x/y/z direction
    !-----------------------------------------------------------------------
    """
    # first linear set
    X,dx = np.linspace(xmin,xmax,nx,retstep=True)
    # control output to screen
    if (control):
        print('%20s %10.2f %10.2f' % ('X.min,X.max [m]:',X.min(),X.max()))
    return X,dx


#================================#
def createFields1D(nx):
    """
    ! function initializes all field with appropriate dimensions
    ! input:
    !  nx              - coordinate increments
    ! output:
    !  K               - conductivity  of node i
    !  S               - storativity of node i
    !  flow ,head      - flow, head of node i
    !  vx              - x-velocity component
    ! use:
    !  K,S,head,flow,vx = libGROWAT1D.createFields1D(nx)
    ! notes:
    """
    K        = np.zeros(nx)
    S        = np.zeros(nx)
    head     = np.zeros(nx)
    flow     = np.zeros(nx)
    vx       = np.zeros(nx)
    return K,S,head,flow,vx


#================================#
def createProperties1D(dataMAT,K,S,X,nx,control=False):
    """
    ! set material properties to first material from materials dict
    ! input:
    !  dataMAT    - dictionary of materials
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    !  X          - x-coordinate [m] of node i
    !  nx         - coordinate increments
    ! output:
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    ! use:
    !  K,S = libGROWAT1D.createProperties1D(dataMAT,K,S,X,nx)
    ! notes:
    """
    # main material (first line in GROWAT2D_materials.in)
    for i in range(nx):
        ij = i
        K[ij] = float(dataMAT[0][3])
        S[ij] = float(dataMAT[0][4])
    # additional materials (other lines in GROWAT2D_materials.in)
    if (dataMAT.shape[1]!=0):
        for ib in range(dataMAT.shape[0]):
            mat = dataMAT[ib][0]
            x1 = float(dataMAT[ib][1])
            x2 = float(dataMAT[ib][2])
            if (x1 < X.min()): sys.exit ('x1 < X.min()')
            if (x2 > X.max()): sys.exit ('x2 > X.max()')

            for i in range(nx):
                ij = i
                if (X[ij] >= x1 and X[ij] <= x2):
                    K[ij] = float(dataMAT[ib][3])
                    S[ij] = float(dataMAT[ib][4])
    if (control):
        print('%20s %10.6f %10.6f' % ('K.min,K.max [m/s]:',K.min(),K.max()))
        print('%20s %10.6f %10.6f' % ('S.min,S.max [1/s]:',S.min(),S.max()))
    return K,S


#================================#
def buildBC1D(dataBC,dx,time,time_scale,head,flow,control=False):
    """
    ! set nodes marked with boundary conditions for current time step
    !  ibound(i)   - boundary flag for node i
    !            0 - unknown head
    !            1 - fixed resurgence
    !            2 - fixed head
    !            3 - fixed recharge
    !            4 - fixed sink
    ! use:
    !  ibound,irecharge,head,flow = libGROWAT1D.buildBC1D(dataBC,dx,time,time_scale,head,flow)
    ! notes:
    """
    nx = head.shape[0]
    ifixhead     = 0; ifixres      = 0
    ifixrecharge = 0; ifixsink     = 0
    # set flow to zero to initialize
    flow = np.zeros(nx)
    # open arrays for boundary index and values
    ibound    = np.zeros(nx,dtype='int')
    irecharge = np.zeros(nx,dtype='int')
    for ib in range(dataBC.shape[0]):
        itype = int(dataBC[ib,0])
        i1 = int(dataBC[ib,1]);i2 = int(dataBC[ib,2])
        t1 = float(dataBC[ib,3]);t2 = float(dataBC[ib,4])
        value = float(dataBC[ib,5])
        #print(i1,i2,t1,t2,value)
        if (i1 >= nx): sys.exit ('i1 > nx-1')
        if (i2 >= nx): sys.exit ('i2 > nx-1')
        # assign values to arrays
        for i in range(i1,i2+1):
            j=0
            k=0
            ijk = i
            surf = dx
            if (i==0 or i==nx-1): surf = 0.5*dx
            # check, if BC is active
            if (time >= t1*time_scale and time <= t2*time_scale):
                # fixed resurgence head node
                if (itype==1):
                    ibound[ijk] = itype
                    head[ijk]   = value
                    ifixres += 1
                # fixed head node
                if (itype==2):
                    ibound[ijk] = itype
                    head[ijk]   = value
                    ifixhead += 1
                # fixed recharge node (mm/timeUnit -> m3/s)
                if (itype==3):
                    irecharge[ijk] = itype
                    flow[ijk]      = value/time_scale/1000.*surf
                    ifixrecharge += 1
                # fixed sink node (m3/timeUnit -> m3/s)
                if (itype==4):
                    ibound[ijk] = itype
                    flow[ijk]      = value/time_scale
                    ifixsink += 1
    if (control):
        print('ifixres:      ',ifixres)
        print('ifixhead:     ',ifixhead)
        print('ifixrecharge: ',ifixrecharge)
        print('ifixsink:     ',ifixsink)
    return ibound,irecharge,head,flow


#================================#
def buildHeadEquations1D_ss(dx,nx,conductm,head,flow,ibound):
    """
    ! function assembles the element entries for the global conductance
    ! matrix and the rhs vector for the steady-state case
    ! input:
    !  dx         - discretisation [m]
    !  nx         - coordinate increments
    !  K          - hydraulic conductivity [m/s]
    !  ibound     - array for boundary markers
    !  head,flow  - head [m] and flow [m3/s] fields
    ! output:
    !  matrix     - global conductivity matrix (sparse)
    !  rhs        -  rhs vector
    ! use:
    !  matrix,rhs = libGROWAT1D.buildHeadEquations1D_ss(dx,nx,K,head,flow,ibound)
    ! notes:
    """
    # initialize fields for sparse matrix and rhs vector
    rhs    = np.zeros([nx])
    matrix = scipy.sparse.lil_array((nx,nx))
    #-----------------------------------------------------------------------
    # assemble matrix, loop over all interior nodes
    #-----------------------------------------------------------------------
    dx2 = 1. / dx**2
    for i in range(nx):
        ij = i
        matrix[ij,ij] = 0.
        # diffusivity in x direction
        if (i != 0 and i != nx-1):
            matrix[ij,ij]   += -2.*dx2*conductm[ij]
            matrix[ij,ij+1] += dx2*conductm[ij+1]
            matrix[ij,ij-1] += dx2*conductm[ij-1]

        if (ibound[ij]==1 or ibound[ij]==2):
            matrix[ij,ij]   = 1.
    #-----------------------------------------------------------------------
    # assemble right-hand side
    #-----------------------------------------------------------------------
    for i in range(nx):
        ij = i
        rhs[ij] = 0.
        # surface inflow
        surface = dx
        if ((i==0) or (i==nx-1)): surface = 0.5*dx
        if (surface != 0.):
            rhs[ij] += -flow[ij]/surface
    #-----------------------------------------------------------------------
    # check for fixed head boundary condition
    #-----------------------------------------------------------------------
    for i in range(nx):
        ij = i
        if (ibound[ij]==1 or ibound[ij]==2):
            matrix[ij,ij] = 10e10*matrix[ij,ij]
            rhs[ij]       = matrix[ij,ij]*head[ij]
    # convert sparse lil format to csr format
    matrix = matrix.tocsr()
    return matrix,rhs


#================================#
def buildHeadEquations1D_t(dx,nx,time_step,conductm,storagem,head,headOld,flow,flowOld,ibound,omega=1.):
    """
    ! function assembles the element entries for the global conductance
    ! matrix and the rhs vector for the transient case
    ! input:
    !  dx         - discretisation [m]
    !  nx         - coordinate increments
    !  K,S        - hydraulic conductivity [m/s], specific storage [1/s]
    !  ibound     - array for boundary markers
    !  head,flow  - head [m] and flow [m3/s] fields
    ! output:
    !  matrix     - global conductivity matrix (sparse)
    !  rhs        -  rhs vector
    ! use:
    !  matrix,rhs = libGROWAT1D.buildHeadEquations1D_ss(dx,nx,K,S,head,headOld,flow,flowOld,ibound)
    ! notes:
    """
    # initialize fields for sparse matrix and rhs vector
    rhs    = np.zeros([nx])
    matrix = scipy.sparse.lil_array((nx,nx))
    #-----------------------------------------------------------------------
    # assemble matrix, loop over all interior nodes
    # omega = 0: fully explicit
    # omega = 1: fully implicit
    #-----------------------------------------------------------------------
    dtdx2 = time_step / dx**2
    for i in range(nx):
        ij = i
        # new unkown head
        matrix[ij,ij] = storagem[ij]
        # diffusivity in x direction (implicit)
        if (i != 0 and i != nx-1):
            matrix[ij,ij]   += 2.*omega*dtdx2*conductm[ij]
            matrix[ij,ij+1] += -omega*dtdx2*conductm[ij+1]
            matrix[ij,ij-1] += -omega*dtdx2*conductm[ij-1]

        if (ibound[ij]==1 or ibound[ij]==2):
            matrix[ij,ij]   = 1.
    #-----------------------------------------------------------------------
    # assemble right-hand side
    #-----------------------------------------------------------------------
    for i in range(nx):
        ij = i
        # old head
        rhs[ij] = headOld[ij]*storagem[ij]
        # surface inflow 
        surface = dx
        if ((i==0) or (i==nx-1)):     surface = 0.5*dx
        if (surface != 0.):
            rhs[ij] += (1.-omega)*flowOld[ij]/surface*time_step 
            rhs[ij] += omega*flow[ij]/surface*time_step
        # diffusivity in x direction (explicit)
        if (i != 0 and i != nx-1):
            rhs[ij] += -2.*(1.-omega)*dtdx2*conductm[ij] 
            rhs[ij] += headOld[ij+1]*(1.-omega)*dtdx2*conductm[ij+1] 
            rhs[ij] += headOld[ij-1]*(1.-omega)*dtdx2*conductm[ij-1]

        if (ibound[ij]==1 or ibound[ij]==2):
            rhs[ij] = headOld[ij]              
    #-----------------------------------------------------------------------
    # check for fixed head boundary condition
    #-----------------------------------------------------------------------
    for i in range(nx):
        ij = i
        if (ibound[ij]==1 or ibound[ij]==2):
            matrix[ij,ij] = 10e10*matrix[ij,ij]
            rhs[ij]       = matrix[ij,ij]*headOld[ij]
    # convert sparse lil format to csr format
    matrix = matrix.tocsr()
    return matrix,rhs


#================================#
def solveHeadEquations1D(matrix,rhs):
    """
    ! Solve linear system of equations
    ! with sparse matrix solver
    ! input:
    !  matrix     - global conductivity matrix (sparse)
    !  rhs        -  rhs vector
    ! output:
    !  head       - head [m] field
    ! use:
    !  head = libGROWAT1D.solveHeadEquations1D(matrix,rhs)
    """
    head = scipy.sparse.linalg.spsolve(matrix,rhs,permc_spec='MMD_AT_PLUS_A')
    return head


#================================#
def solveVelocities1D(nx,time_scale,x,K,head):
    """
    ! function calculates velocity components in center of block
    ! input:
    !  nx                - coordinate increments
    !  time_scale        - time scale to convert velocities to user timescale
    !  x                 - x coordinate of node i
    !  K                 - conductivity  of node i
    !  head              - head of node i
    ! output:
    !  xc                - x coordinate of velocity node j
    !  vcx               - x velocity component of velocity node j
    ! use:
    !  xc,vcx = libGROWAT1D.solveVelocities1D(nx,time_scale,x,K,head)
    """
    # define velocity components in center of grid
    nvel     = (nx-1)
    xc       = np.zeros(nvel)
    vcx      = np.zeros(nvel)
    iv = 0
    for i in range(nx-1):
        ij = i
        xc[iv]  = 0.5*(x[ij+1]+x[ij])
        Kave    = 0.5*(K[ij] + K[ij+1])
        vcx[iv] = -Kave * (head[ij+1]-head[ij])/(x[ij+1]-x[ij])
        vcx[iv] *= time_scale
        iv = iv+1
    return xc,vcx


#================================#
def saveToScreen1D(saved,time,time_scale,head,vabs):
    """
    function creates on-line screen output
    """
    format = "%3s t:%10i h: %8.2f +/- %8.2f %8.2f - %8.2f v: %8.2f - %8.2f"
    headMean,headStd = round(head.mean(),2),round(head.std(),2)
    headMin,headMax  = round(head.min(),2),round(head.max(),2)
    vabsMin,vabsMax  = round(vabs.min(),2),round(vabs.max(),2)
    print(format % (saved,time/time_scale,headMean,headStd,headMin,headMax,vabsMin,vabsMax)) 
    return
        
        
#================================#
def saveHeadsAndVelocities1D(itime,time,time_scale,whichtime,x,head,flow,xc,vcx,ibound,irecharge,name='FD_'):
    """
    ! function saves head, flow and velocity components to file
    ! input:
    !  itime,time        - time increment and current time [s]
    !  whichtime         - time flag
    !  time_scale        - time scale to convert velocities to user timescale
    !  x                 - x coordinate of node i
    !  xc                - x coordinate of velocity node j
    !  vcx               - x velocity component of velocity node j
    !  K                 - conductivity [m/s] of node i
    !  head              - head [m] of node i
    !  flow              - flow [m3/s] of node i
    !  ibound            - flag for boundary nodes
    !  irecharge         - flag for recharge nodes
    ! output:
    !  (to files)
    ! use:
    !  libGROWAT1D.saveHeadsAndVelocities1D(itime,time,time_scale,whichtime,x,head,flow,xc,vcx,ibound,irecharge)
    """
    # save heads and flow to filename1
    format1 = "%10i %12.2f %12.2f %12.2f %2i %2i"
    filename1 = "work/"+name+f"{itime:04}.heads"
    f = open(filename1,'w')
    print('time,whichtime: ',time/time_scale,whichtime,file=f)
    for i in range(x.shape[0]):
        print(format1 % (i,x[i],head[i],flow[i],ibound[i],irecharge[i]),file=f)
    f.close()
    
    # save velocities to filename2
    format2 = "%10i %12.2f %12.2f"
    filename2 = "work/"+name+f"{itime:04}.vel"
    f = open(filename2,'w')
    print('time,whichtime: ',time/time_scale,whichtime,file=f)
    for i in range(xc.shape[0]):
        print(format2 % (i,xc[i],vcx[i]),file=f)
    f.close()
    return


#================================#
def plotHeads1D(itime,time,time_scale,x,xc,head,vcx,ibound,irecharge,vmin=0.,vmax=4.,vstep=21,plot=False,name='FD_'):
    filename = "work/"+name+f"{itime:04}.png"
    cmap = mpl.cm.jet
    levels = np.linspace(0,1,21,endpoint=True)
    norm   = mpl.colors.BoundaryNorm(levels, cmap.N)
    plt.figure(figsize=(12,5))
    plt.title('Time: '+str(round(time/time_scale,0))+' d')
    plt.xlabel('x [m]')
    plt.ylabel('head [m]')
    plt.ylim([vmin,vmax])
    plt.fill_between(x,head,0.,label='head',lw=3,alpha=0.4)
    plt.plot(x[ibound==1],head[ibound==1],lw=0,marker='s',markersize=10,alpha=0.6,label='1')
    plt.plot(x[ibound==2],head[ibound==2],lw=0,marker='s',markersize=10,alpha=0.6,label='2')
    plt.plot(x[irecharge==3],4*np.ones_like(x)[irecharge==3],lw=0,marker='o',markersize=10,alpha=0.6,label='3')
    plt.plot(x[ibound==4],np.ones_like(x)[ibound==4],lw=0,marker='^',markersize=10,alpha=0.6,label='4')
    vcabs = np.sqrt(vcx**2)
    cbar1=plt.quiver(xc,np.ones_like(xc),vcx/vcabs,np.zeros_like(vcx),vcabs,alpha=0.4,width=0.005,scale=20,pivot="middle",norm=norm,cmap=cmap)
    clabel1=plt.colorbar(cbar1,extend='both')
    clabel1.set_label('v [m/d]')
    plt.legend(loc='center left')
    plt.tight_layout()
    plt.savefig(filename)
    if (not plot):
        plt.close()
    return
    
