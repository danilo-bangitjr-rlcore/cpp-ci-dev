C FILE: CALLBACK.F
      SUBROUTINE FOO(FUN,R)
      EXTERNAL FUN
      INTEGER I
      DOUBLE PRECISION R
      DOUBLE PRECISION O(2)
Cf2py intent(out) r
      R = 1
      DO I=-5,5
         CALL FUN(I, O)
         R = R + O(2)
      ENDDO
      END
C END OF FILE CALLBACK.F
