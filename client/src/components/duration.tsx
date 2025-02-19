import React, { useState } from "react";
import { Temporal } from "temporal-polyfill";
import { Field, FieldGroup, Label } from "./fieldset";
import { Input } from "./input";

interface DurationInputProps {
  onChange: (isoDuration: string) => void;
  value?: string;
  units?: (keyof Temporal.DurationLike)[];
}

const DurationInput: React.FC<DurationInputProps> = ({
  onChange,
  value,
  units = ["hours", "minutes", "seconds"],
}) => {
  const rawDuration = value
    ? Temporal.Duration.from(value)
    : new Temporal.Duration();
  const [durationParts, setDurationParts] = useState<Temporal.DurationLike>({
    years: rawDuration.years,
    months: rawDuration.months,
    weeks: rawDuration.weeks,
    days: rawDuration.days,
    hours: rawDuration.hours,
    minutes: rawDuration.minutes,
    seconds: rawDuration.seconds,
    milliseconds: rawDuration.milliseconds,
    microseconds: rawDuration.microseconds,
  });

  const handleChange = (value: string, unit: keyof Temporal.DurationLike) => {
    const parsedValue = parseInt(value, 10);
    if (!isNaN(parsedValue) && parsedValue >= 0) {
      updateDuration(unit, parsedValue);
    }
  };

  const updateDuration = (unit: keyof Temporal.DurationLike, value: number) => {
    setDurationParts({ ...durationParts, [unit]: value });
    const rawDuration = Temporal.Duration.from({
      ...durationParts,
      [unit]: value,
    });
    onChange(rawDuration.toString());
  };

  const capitalizeFirstLetter = (val: string) =>
    String(val).charAt(0).toUpperCase() + String(val).slice(1);

  return (
    <FieldGroup className="grid grid-cols-3 grid-gap-2">
      {units.map((unit) => (
        <Field className="flex flex-col mr-2" key={unit}>
          <Label>{capitalizeFirstLetter(unit)}</Label>
          <Input
            name={unit}
            type="number"
            value={rawDuration[unit]}
            onChange={(e) => handleChange(e.target.value, unit)}
            min="0"
          />
        </Field>
      ))}
    </FieldGroup>
  );
};

export default DurationInput;
