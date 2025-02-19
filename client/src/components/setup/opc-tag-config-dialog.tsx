import {
  Dialog,
  DialogActions,
  DialogBody,
  DialogTitle,
} from "../../components/dialog";
import {
  Description,
  Field,
  FieldGroup,
  Fieldset,
  Label,
} from "../../components/fieldset";
import { Input } from "../../components/input";
import { Button } from "../../components/button";
import { components } from "../../api-schema";
import {
  ChangeEventHandler,
  FormEventHandler,
  MouseEventHandler,
  useEffect,
  useState,
} from "react";
import { Checkbox, CheckboxField } from "../checkbox";
import { DeepPartial } from "../../utils/main-config";
import { Text } from "../text";

interface TagConfigProps {
  dialogOpen: boolean;
  setDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
  selectedTagConfig?: DeepPartial<components["schemas"]["TagConfig"]>;
  tagConfigIndex?: number;
  handleSubmittedOPCNodeTagConfig: (
    updatedNode: components["schemas"]["TagConfig"],
  ) => void;
  handleDeleteOPCNodeTagConfig: (tagConfigIndex: number) => void;
}

export const TagConfigDialog = ({
  dialogOpen,
  setDialogOpen,
  selectedTagConfig,
  tagConfigIndex,
  handleSubmittedOPCNodeTagConfig,
  handleDeleteOPCNodeTagConfig,
}: TagConfigProps) => {
  const [nodeIdentifier, setNodeIdentifier] = useState<string>("");
  const [name, setName] = useState<string>("");
  const [isSetpoint, setIsSetpoint] = useState<boolean>(false);
  const [operatingRangeLow, setOperatingRangeLow] = useState<number | "">("");
  const [operatingRangeHigh, setOperatingRangeHigh] = useState<number | "">("");
  const [redBoundsLow, setRedBoundsLow] = useState<number | "">("");
  const [redBoundsHigh, setRedBoundsHigh] = useState<number | "">("");
  const [yellowBoundsLow, setYellowBoundsLow] = useState<number | "">("");
  const [yellowBoundsHigh, setYellowBoundsHigh] = useState<number | "">("");

  useEffect(() => {
    setNodeIdentifier(`${selectedTagConfig?.node_identifier ?? ""}`);
    setName(selectedTagConfig?.name ?? "");

    // action_constructor to isSetpoint boolean
    const lenAc = selectedTagConfig?.action_constructor?.length ?? 0;
    setIsSetpoint(lenAc > 0);

    setOperatingRangeLow(selectedTagConfig?.operating_range?.[0] ?? "");
    setOperatingRangeHigh(selectedTagConfig?.operating_range?.[1] ?? "");
    setYellowBoundsLow(selectedTagConfig?.yellow_bounds?.[0] ?? "");
    setYellowBoundsHigh(selectedTagConfig?.yellow_bounds?.[1] ?? "");
    setRedBoundsLow(selectedTagConfig?.red_bounds?.[0] ?? "");
    setRedBoundsHigh(selectedTagConfig?.red_bounds?.[1] ?? "");
  }, [selectedTagConfig]);

  const setNumberWithUndefined: (
    chevt: React.Dispatch<React.SetStateAction<number | "">>,
  ) => ChangeEventHandler<HTMLInputElement> = (chevt) => (e) => {
    const val = e.target.value;
    if (val) {
      chevt(Number(e.target.value));
    } else {
      chevt("");
    }
  };

  const handleSubmitOPCNodeTagConfig: FormEventHandler<HTMLFormElement> = (
    e,
  ) => {
    e.preventDefault();

    const createdTagConfig: components["schemas"]["TagConfig"] = {
      agg: "avg",
      is_endogenous: true,
      is_meta: false,
      node_identifier: nodeIdentifier,
      name,
      action_constructor: isSetpoint
        ? [
            {
              name: "identity",
            },
          ]
        : [],
      state_constructor: !isSetpoint
        ? [
            {
              name: "multi_trace",
              trace_values: [0, 0.9],
            },
          ]
        : [],
      operating_range: [
        typeof operatingRangeLow === "number" ? operatingRangeLow : null,
        typeof operatingRangeHigh === "number" ? operatingRangeHigh : null,
      ],
      yellow_bounds: [
        typeof yellowBoundsLow === "number" ? yellowBoundsLow : null,
        typeof yellowBoundsHigh === "number" ? yellowBoundsHigh : null,
      ],
      red_bounds: [
        typeof redBoundsLow === "number" ? redBoundsLow : null,
        typeof redBoundsHigh === "number" ? redBoundsHigh : null,
      ],
    };
    handleSubmittedOPCNodeTagConfig(createdTagConfig);
    setDialogOpen(false);
  };

  const handleDeleteNode: MouseEventHandler<HTMLButtonElement> = () => {
    if (tagConfigIndex !== undefined) {
      handleDeleteOPCNodeTagConfig(tagConfigIndex);
    }
    setDialogOpen(false);
  };

  return (
    <Dialog size="xl" open={dialogOpen} onClose={setDialogOpen}>
      <DialogTitle>Configure Tag</DialogTitle>
      <form onSubmit={handleSubmitOPCNodeTagConfig}>
        <DialogBody>
          <Fieldset>
            <FieldGroup>
              <Field className="mb-2">
                <Label htmlFor="node_identifier">OPC Node ID</Label>
                <Input
                  className="mb-0"
                  id="node_identifier"
                  name="node_identifier"
                  type="text"
                  value={nodeIdentifier}
                  onChange={(e) => setNodeIdentifier(e.target.value)}
                  autoFocus={true}
                />
                <Text className="">Full OPC-UA Node Identifier.</Text>
              </Field>
              <Field className="mb-2">
                <Label>Name</Label>
                <Input
                  name="tag_name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
                <Text className="">Tag Configuration name.</Text>
              </Field>
            </FieldGroup>
            <div className="border-gray-200 border-2 rounded-sm p-1 mt-1 mb-1">
              <FieldGroup className="flex flex-row justify-between content-between">
                <Field className="flex flex-col mb-0">
                  <Label htmlFor="operating_range_low">
                    Operating Range Low
                  </Label>
                  <Input
                    className="mb-0"
                    id="operating_range_low"
                    name="operating_range_low"
                    type="number"
                    value={operatingRangeLow}
                    onChange={setNumberWithUndefined(setOperatingRangeLow)}
                  />
                </Field>
                <Field className="flex flex-col mb-0">
                  <Label htmlFor="operating_range_high">
                    Operating Range High
                  </Label>
                  <Input
                    className="mb-0"
                    id="operating_range_high"
                    name="operating_range_high"
                    type="number"
                    value={operatingRangeHigh}
                    onChange={setNumberWithUndefined(setOperatingRangeHigh)}
                  />
                </Field>
              </FieldGroup>
              <Description>
                Absolute range within which a process or equipment can function.
              </Description>
            </div>
            <div className="border-rose-200 border-2 rounded-sm p-1 mt-1 mb-1">
              <FieldGroup className="flex flex-row justify-between content-between">
                <Field className="flex flex-col mb-0">
                  <Label htmlFor="red_bounds_low">Red Bounds Low</Label>
                  <Input
                    className="mb-0"
                    id="red_bounds_low"
                    name="red_bounds_low"
                    type="number"
                    value={redBoundsLow}
                    onChange={setNumberWithUndefined(setRedBoundsLow)}
                  />
                </Field>
                <Field className="flex flex-col mb-0">
                  <Label htmlFor="red_bounds_high">Red Bounds High</Label>
                  <Input
                    className="mb-0"
                    id="red_bounds_high"
                    name="red_bounds_high"
                    type="number"
                    value={redBoundsHigh}
                    onChange={setNumberWithUndefined(setRedBoundsHigh)}
                  />
                </Field>
              </FieldGroup>
              <Description>
                Critical limits that indicate a risk of failure, damage, or
                unsafe conditions.
              </Description>
            </div>
            <div className="border-amber-200 border-2 rounded-sm p-1 mt-1 mb-1">
              <FieldGroup className="flex flex-row justify-between content-between">
                <Field className="flex flex-col mb-0">
                  <Label htmlFor="yellow_bounds_low">Yellow Bounds Low</Label>
                  <Input
                    className="mb-0"
                    id="yellow_bounds_low"
                    name="yellow_bounds_low"
                    type="number"
                    value={yellowBoundsLow}
                    onChange={setNumberWithUndefined(setYellowBoundsLow)}
                  />
                </Field>
                <Field className="flex flex-col mb-0">
                  <Label htmlFor="yellow_bounds_high">Yellow Bounds High</Label>
                  <Input
                    className="mb-0"
                    id="yellow_bounds_high"
                    name="yellow_bounds_high"
                    type="number"
                    value={yellowBoundsHigh}
                    onChange={setNumberWithUndefined(setYellowBoundsHigh)}
                  />
                </Field>
              </FieldGroup>
              <Description>
                Warning limits that signal deviations from the optimal operating
                range, requiring attention to prevent escalation to red bounds.
              </Description>
            </div>
          </Fieldset>
          <CheckboxField>
            <Checkbox
              name="is_setpoint"
              checked={isSetpoint}
              onChange={setIsSetpoint}
            />
            <Label>Is Setpoint</Label>
            <Description>
              Specify that this OPC Node represents a plant tunable setpoint.
            </Description>
          </CheckboxField>
        </DialogBody>
        <DialogActions className="">
          <Button plain onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          {tagConfigIndex !== undefined ? (
            <>
              <Button color="red" onClick={handleDeleteNode}>
                Delete
              </Button>
              <Button type="submit">Save</Button>
            </>
          ) : (
            <Button type="submit">Add</Button>
          )}
        </DialogActions>
      </form>
    </Dialog>
  );
};
