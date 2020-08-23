
      SUBROUTINE ROLLOUT( &
            CON,INIT_X,TARG_X, &
            OBS,OBS_N, &
            OBS_R_FRAC, &
            X_LEN, CON_LEN, &
            N,TS, &
            COST &
         )
      IMPLICIT NONE
!
!     COMPUTE COST BY ITERATIVE ROLLOUT
!
      REAL(8) CON(CON_LEN)
      REAL(8) INIT_X(X_LEN)
      REAL(8) TARG_X(X_LEN-1)
      REAL(8) STEP_X(X_LEN)
      REAL(8) STEP_X_WO_A(X_LEN-1)
      REAL OBS(OBS_N, X_LEN-1)
      
      REAL(8) :: COST
      REAL(8) :: NORM
      REAL(8) :: OBS_MIN_DIST
      REAL(8) :: OBS_R
      REAL :: OBS_R_FRAC

      INTEGER :: X_LEN
      INTEGER :: CON_LEN
      INTEGER :: OBS_N

      REAL(8), dimension(X_LEN,CON_LEN) :: J
      INTEGER, OPTIONAL :: N
      REAL, OPTIONAL :: TS

      integer :: I

!f2py intent(in) obs_n
!f2py intent(in) obs
!f2py intent(in) X_LEN
!f2py intent(in) CON_LEN
!f2py intent(in) init_x
!f2py intent(in) targ_x
!f2py intent(in) con
!f2py intent(in) n
!f2py intent(out) cost
!f2py real :: ts = .1
!f2py integer :: n = 50
!f2py real :: OBS_R_FRAC = 15.

!     ASSIGN STEP FOR X
      STEP_X = INIT_X
 
      DO I=1,N
         call JACOBIAN_VEHICLE(STEP_X, J)
         ! PRINT *, STEP_X
         STEP_X = STEP_X + &
            MATMUL(J, CON) * TS
         STEP_X_WO_A=STEP_X(1:X_LEN-1)
         NORM = NORM2( &
               STEP_X_WO_A - TARG_X &
               )

         IF (NORM < .1) THEN
            EXIT
         END IF
         
         OBS_MIN_DIST = MINVAL( &
            NORM2(spread(STEP_X_WO_A, 1, OBS_N) - OBS, 2) &
            )
         OBS_R = OBS_R_FRAC / &
            (TS * I * OBS_MIN_DIST)

         COST = COST + NORM + OBS_R
      ENDDO
      
      END


      SUBROUTINE JACOBIAN_VEHICLE(X, R)
      IMPLICIT NONE

      REAL(8), dimension(3) :: X
      REAL(8), dimension(3, 2) :: R
!f2py intent(in) x
!f2py intent(out) r
      
      R = 0.0

!     FORTRAN MATRIX INDICES ORDER ARE REVERSED
      R(1,1) = COS(X(3))
      R(2,1) = SIN(X(3))
      R(3,2) = 1.

      END