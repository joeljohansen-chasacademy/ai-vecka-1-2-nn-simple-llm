// To run this code you need to install the following dependencies:
// npm install @google/genai mime
// npm install -D @types/node

import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
dotenv.config();

async function main() {
	const ai = new GoogleGenAI({
		apiKey: process.env.GEMINI_API_KEY,
	});
	const config = {
		temperature: 0,
		thinkingConfig: {
			thinkingBudget: 0,
		},
		systemInstruction: [
			{
				text: `Du jobbar på Chas Academy och ska svara vänligt på alla frågor som inkommer. Svara alltid på svenska.`,
			},
		],
	};
	const model = "gemini-flash-lite-latest";
	const contents = [
		{
			role: "user",
			parts: [
				{
					text: `Hej när ska lämna in uppgift 2?`,
				},
			],
		},
	];
	/* 
	const response = await ai.models.generateContentStream({
		model,
		config,
		contents,
	});

	let finalResponse = "";

	for await (const chunk of response) {
		console.log(chunk.text);
		finalResponse += chunk.text;
	}

	console.log(finalResponse); */

	const response = await ai.models.generateContent({
		model,
		config,
		contents,
	});

	console.log(response);
	console.log(response.text);
}

main();
