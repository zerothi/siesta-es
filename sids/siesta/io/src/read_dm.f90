subroutine read_dm(fname,no_u,nspin,maxnd, &
     numd,listdptr,listd,DM)

  ! Precision 
  integer, parameter :: dp = selected_real_kind(p=15)

  ! Input parameters
  character(len=*) :: fname
  integer :: no_u, nspin, maxnd
  integer :: listd(maxnd), numd(no_u), listdptr(no_u)
  real(dp) :: DM(maxnd,nspin)
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(in) :: no_u, nspin, maxnd
!f2py integer, intent(out), dimension(no_u)  :: numd, listdptr
!f2py integer, intent(out), dimension(maxnd) :: listd
!f2py real*8, intent(out), dimension(maxnd,nspin) :: DM

! Internal variables and arrays
  integer :: iu
  integer :: is, id, ip

  ! Local readables
  integer :: lno_u, lnspin

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) lno_u, lnspin
  if ( lno_u /= no_u ) stop 'Error in reading data, not allocated, no_u'
  if ( lnspin /= nspin ) stop 'Error in reading data, not allocated, nspin'

  read(iu) numd

! Create listdptr
  listdptr(1) = 0
  do id = 2 , no_u
     listdptr(id) = listdptr(id-1) + numd(id-1)
  end do

! Read listd
  do id = 1 , no_u
     ip = listdptr(id)
     read(iu) listd(ip+1:ip+numd(id))
  end do

! Read Density matrix
  do is = 1 , nspin
     do id = 1 , no_u
        ip = listdptr(id)
        read(iu) DM(ip+1:ip+numd(id),is)
     end do
  end do

  close(iu)

end subroutine read_dm
