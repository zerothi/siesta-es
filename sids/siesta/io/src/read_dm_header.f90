subroutine read_dm_header(fname,nspin,no_u,maxnd)

  ! Input parameters
  character(len=*) :: fname
  integer :: no_u, nspin, maxnd
! Define f2py intents
!f2py intent(in)  :: fname
!f2py intent(out) :: no_u, nspin, maxnd

! Internal variables and arrays
  integer :: iu

  integer, allocatable :: numd(:)

  iu = 1804
  open(iu,file=trim(fname),status='old',form='unformatted')

  read(iu) no_u, nspin
  allocate(numd(no_u))
  read(iu) numd
  maxnd = sum(numd)
  deallocate(numd)

  close(iu)

end subroutine read_dm_header
