import Navbar from "@/components/navigation/Navbar";
import LanguageCard from "./LanguageCard";
import { createContext, useEffect, useState } from "react";
import { apiUrl } from "@/main";

export type Scenario = {
  scenario_id: number;
  name: string;
  image: string;
  context: string;
  first_message: string;
};

export const FetchScenariosContext = createContext(() => {});

export default function HomePage() {
  const user = JSON.parse(localStorage.getItem("SpeakEasyUser") as string);
  const user_id = user["uid"];
  const [cardData, setCardData] = useState<Scenario[]>([]);

  async function fetchScenarios(): Promise<void> {
    try {
      const response = await fetch(`${apiUrl}/api/v1/scenarios/${user_id}`);
      const json = await response.json();
      const scenarios: Scenario[] = json["scenarios"];
      setCardData(scenarios);
    } catch (err) {
      console.error("Error fetching scenarios: ", err);
    }
  }

  useEffect(() => {
    fetchScenarios();
  }, []);

  return (
    <FetchScenariosContext.Provider value={fetchScenarios}>
      <div className="min-h-screen w-screen">
        <Navbar />
        <div className="flex flex-col space-y-8 items-start p-4 w-full">
          <div className="flex flex-col w-full space-y-6 items-start">
            <h1 className="text-3xl font-bold">My Scenarios</h1>
            <div className="flex space-x-4 w-full overflow-auto pb-4">
              {cardData.map((scenario) => (
                <LanguageCard key={scenario.scenario_id} scenario={scenario} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </FetchScenariosContext.Provider>
  );
}
