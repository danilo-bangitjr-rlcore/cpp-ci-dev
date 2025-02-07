import { Button } from "../button";
import { MainConfigStepsContext } from "../../utils/main-config";
import { useContext } from "react";

export const SetupConfigNav = () => {
  const { nextStep, prevStep } = useContext(MainConfigStepsContext);
  return (
    <span className="isolate inline-flex rounded-md shadow-xs">
      {prevStep && (
        <Button className="mr-1" to={prevStep}>
          Back
        </Button>
      )}
      {nextStep && <Button to={nextStep}>Next</Button>}
    </span>
  );
};
