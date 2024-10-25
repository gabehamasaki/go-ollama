package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/prompts"
)

func main() {

	llm, err := ollama.New(ollama.WithModel("llama3.2"))
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	ctx := context.Background()

	chatPromptTemplate := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate("You are a knowledgeable AI assistant specializing in video games. Provide accurate and detailed information.", []string{"input"}),
		prompts.NewSystemMessagePromptTemplate("If the inquiry does not pertain to video games, respond with 'I'm sorry, I don't have that information.'", []string{"input"}),
		prompts.NewSystemMessagePromptTemplate("If the question is unrelated to video games, reply with 'I'm sorry, I don't have that information.'", []string{"input"}),
		prompts.NewSystemMessagePromptTemplate("When asked about video games, provide comprehensive and detailed responses.", []string{"input"}),
		prompts.NewSystemMessagePromptTemplate("Ensure the response is delivered in the language specified: {{.language}}.", []string{"language"}),
		prompts.NewHumanMessagePromptTemplate("User: {{.input}}", []string{"input"}),
	})

	result, err := chatPromptTemplate.Format(map[string]any{"input": "What is the best video game of all time?", "language": "English"})
	if err != nil {
		log.Fatalf("Failed to format messages: %v", err)
	}

	_, err = llm.Call(ctx, result,
		llms.WithTemperature(0.9),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}),
	)
	if err != nil {
		log.Fatalf("Failed to get completion: %v", err)
	}

}
