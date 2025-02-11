import {
  Dialog,
  DialogActions,
  DialogBody,
  DialogDescription,
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

interface TagConfigProps {
  dialogOpen: boolean;
  setDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
  selectedNode?: components["schemas"]["OpcNodeDetail"] | undefined;
  selectedTagConfig?:
    | DeepPartial<components["schemas"]["TagConfig"]>
    | undefined;
  handleSubmittedOPCNodeTagConfig: (
    updatedNode: components["schemas"]["TagConfig"],
  ) => void;
  handleDeleteOPCNodeTagConfig: (nodeIdentifier: string | number) => void;
}

export const TagConfigDialog = ({
  dialogOpen,
  setDialogOpen,
  selectedNode,
  selectedTagConfig,
  handleSubmittedOPCNodeTagConfig,
  handleDeleteOPCNodeTagConfig,
}: TagConfigProps) => {
  const nodeIdentifier =
    selectedNode?.nodeid ?? selectedTagConfig?.node_identifier ?? "";

  const [name, setName] = useState<string>("");
  const [isSetpoint, setIsSetpoint] = useState<boolean>(false);
  const [operatingRangeLow, setOperatingRangeLow] = useState<number | "">("");
  const [operatingRangeHigh, setOperatingRangeHigh] = useState<number | "">("");
  const [redBoundsLow, setRedBoundsLow] = useState<number | "">("");
  const [redBoundsHigh, setRedBoundsHigh] = useState<number | "">("");
  const [yellowBoundsLow, setYellowBoundsLow] = useState<number | "">("");
  const [yellowBoundsHigh, setYellowBoundsHigh] = useState<number | "">("");

  useEffect(() => {
    setName(selectedTagConfig?.name ?? selectedNode?.key ?? "");

    // action_constructor to isSetpoint boolean
    const lenAc = selectedTagConfig?.action_constructor?.length ?? 0;
    setIsSetpoint(
      lenAc > 0 || (selectedNode?.key.toLowerCase().endsWith("sp") ?? false),
    );

    setOperatingRangeLow(selectedTagConfig?.operating_range?.[0] ?? "");
    setOperatingRangeHigh(selectedTagConfig?.operating_range?.[1] ?? "");
    setYellowBoundsLow(selectedTagConfig?.yellow_bounds?.[0] ?? "");
    setYellowBoundsHigh(selectedTagConfig?.yellow_bounds?.[1] ?? "");
    setRedBoundsLow(selectedTagConfig?.red_bounds?.[0] ?? "");
    setRedBoundsHigh(selectedTagConfig?.red_bounds?.[1] ?? "");
  }, [selectedNode, selectedTagConfig]);

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
    handleDeleteOPCNodeTagConfig(nodeIdentifier);
    setDialogOpen(false);
  };

  return (
    <Dialog size="xl" open={dialogOpen} onClose={setDialogOpen}>
      <DialogTitle>Configure Tag</DialogTitle>
      <DialogDescription>Configure bounds for this tag.</DialogDescription>
      <form onSubmit={handleSubmitOPCNodeTagConfig}>
        <DialogBody>
          <Fieldset>
            <FieldGroup className="mb-2">
              <Field className="mb-2">
                <Label htmlFor="node_identifier">OPC Node ID</Label>
                <Input
                  className="mb-0"
                  id="node_identifier"
                  name="node_identifier"
                  type="text"
                  disabled={true}
                  value={nodeIdentifier}
                />
              </Field>
              <Field className="mb-2">
                <Label>Name</Label>
                <Input
                  name="tag_name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </Field>
            </FieldGroup>
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
            <FieldGroup className="flex flex-row justify-between content-between">
              <Field className="flex flex-col mb-0">
                <Label htmlFor="operating_range_low">Operating Range Low</Label>
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
          </Fieldset>
        </DialogBody>
        <DialogActions>
          <Button plain onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          {selectedTagConfig && (
            <Button color="red" onClick={handleDeleteNode}>
              Delete
            </Button>
          )}
          <Button type="submit">Save</Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};
