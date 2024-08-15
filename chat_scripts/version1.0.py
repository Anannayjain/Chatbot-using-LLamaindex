# !pip install llama_index
# !pip install llama-index-experimental
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
import pandas as pd
import os
os.environ["OPENAI_API_KEY"] = [YOUR_OPENAI_API_KEY]

# Make sure that date is alerady in yyyy-mm-dd format

class SearchQueryPipeline:
    def __init__(self, df, model="gpt-3.5-turbo"):
        # self.csv_path = csv_path
        self.df = df
        self.llm = OpenAI(model=model) # max-tokens
        self.instruction_str = (
            "0. No need to use pd.to_datetime as 'Date' column is already in date_time format.\n"
            "1. Convert the query to executable Python code using Pandas.\n"
            "2. The Date column is already converted to Datetime Format using expression "
            "3. If asked about keywords, be careful because all keywords in Search keyword are not unique."
            "4. For queries related to ROI or Keywords, calculate the total cost and total clicks for each unique keyword before computing ROI or related metrics.\n"
              "example : <query> Best keyword in terms of roi, \n"
              "<response> df.groupby('Search keyword').agg({'Clicks': 'sum', 'Cost': 'sum'}).assign(ROI=lambda x: x['Clicks'] / x['Cost']).sort_values('ROI', ascending=False).index[0] \n"
            "5. Think about problem, break in steps for example:"
              "example : <query> What was the average number of clicks per day?, \n"
              "<response> df.groupby('Date')['Clicks'].sum().mean() \n"
              "First calcuated total clicks for each day, and then taken average. \n"
            "6. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
            "7. The code should represent a solution to the query. \n"
            "8. PRINT ONLY THE EXPRESSION.\n"
            "9. Do not quote the expression.\n"
        )

        self.pandas_prompt_str = (
            "You are working with a pandas dataframe in Python.\n"
            "The name of the dataframe is `df`.\n"
            "The dataframe has nine columns: \n"
            "1. `Date`: The date associated with each search keyword entry. \n"
            "2. `Campaign_ID`: The unique identifier for each campaign. This have only 59 unique values in all records. Prefer working with this if asked about campaign\n"
            "3. `Campaign_name`: This columns have only two values 'Local Search & Retargeting Display'\
                                  'Local Search & Retargeting Display_NB'.  \n"
            "4. `Campaign_Type`: This columns has two value <program> or <campaign>. `campaign` means that it is related to `hiring` \
                                  and program means that it is relted to `Local Search & Retargeting Display`  \n"
            "5. `Search keyword`: The keywords used in the search queries.\n"
            "6. `Impression`: The number of impressions for the corresponding search keyword.\n"
            "7. `Clicks`: The number of clicks for the corresponding search keyword.\n"
            "8. `Currency code`: The currency used for cost column, It has a single value USD.\n"
            "9. `Cost`: The cost involved in USD for the corresponding search keyword.\n"
            "This is the result of `print(df.head())`:\n"
            "{df_str}\n\n"
            "Follow these instructions:\n"
            "{instruction_str}\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )

        self.response_synthesis_prompt_str = (
            "Given an input question, synthesize a response from the query results.\n"
            "Query: {query_str}\n\n"
            "Pandas Instructions: \n{pandas_instructions}\n\n"
            "Pandas Output: {pandas_output}\n\n"
            "Response: "
        )

        self.pandas_prompt = PromptTemplate(self.pandas_prompt_str).partial_format(
            instruction_str=self.instruction_str, df_str=self.df.head(5)
        )

        self.pandas_output_parser = PandasInstructionParser(self.df)
        self.response_synthesis_prompt = PromptTemplate(self.response_synthesis_prompt_str)

        self.qp = QP(
            modules={
                "input": InputComponent(),
                "pandas_prompt": self.pandas_prompt,
                "text2pandas_llm": self.llm,
                "pandas_output_parser": self.pandas_output_parser,
                "response_synthesis_prompt": self.response_synthesis_prompt,
                "response_synthesis_llm": self.llm,
            },
            verbose=True,
        )
        self.qp.add_chain(["input", "pandas_prompt", "text2pandas_llm", "pandas_output_parser"])
        self.qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        self.qp.add_link("text2pandas_llm", "response_synthesis_prompt", dest_key="pandas_instructions")
        self.qp.add_link("pandas_output_parser","response_synthesis_prompt",dest_key="pandas_output")
        self.qp.add_link("response_synthesis_prompt", "response_synthesis_llm")


    def query_response(self, question):
        response, intermediates = self.qp.run_with_intermediates(
        query_str= question, )

        print(intermediates['pandas_output_parser'].inputs['input'].message.content)
        return response.message.content

class AdsQueryPipeline:
    def __init__(self, df, model="gpt-3.5-turbo"):
        # self.csv_path = csv_path
        self.df = df
        self.llm = OpenAI(model='gpt-4o',temperature=0.0) # max-tokens 'gpt-4o'
        self.instruction_str = (
            "0. No need to use pd.to_datetime as 'Date' column is already in date_time format.\n"
            "1. Convert the query to executable Python code using Pandas.\n"
            "2. The Date column is already converted to Datetime Format using expression \n"
            "3. If asked about keywords, be careful because all keywords in Search keyword are not unique. \n"
            "4. If asked about `Brand Video` then fetch results for both `Brand Video` and `Brand Video Full` \n"

            "5. For the questions related to efficiency, effectiveness of ads or programs, consider all ther parmeters like `Reach`,`Impressions`,`Adds to cart`,`Unique link clicks`,`Amount Spent HYP 20` for Answering, if any specific parameter is not mentioned in question. \n"
            "6. Think about problem, break in steps \n"
            "7. Decide if you need to use groupby for questions \n"
                "example : <query>Which location have zero reach \n"
                "first find df.groupby('Location').agg({'Reach': 'sum'}) and then answer the question \n"
            "8. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
            "9. The code should represent a solution to the query. \n"
            "10. PRINT ONLY THE EXPRESSION.\n"
            "11. Do not quote the expression.\n"
        )

        self.pandas_prompt_str = (
            "You are working with a pandas dataframe in Python.\n"
            "The name of the dataframe is `df`.\n"
            "The dataframe has Thirteen columns: \n"
            "1. `LocationID`: A unique identifier for the location, e.g., '38QSR44'. \n"
            "2. `Location`: The name of the location, e.g., 'Austin'.\n"
            "3. `AdProgram`: The name of the advertising program, e.g., 'JINYA_Program_Interests'.\n"
            "4. `Ad_ID`: The unique identifier for the advertisement, e.g., 'IGDC4847'.  \n"
            "5. `Ad_description`: The description of the advertisement, e.g., 'Brand Video Full'.\n"
            "6. `Date`: The date and time when the data was recorded, formatted as YYYY-MM-DD, e.g., '2022-01-01'\n"
            "7. `Campaign_ID`: The unique identifier for the campaign, e.g., 'WBUI5873'.\n"
            "8. `Campaign Description`: The description of the campaign, e.g., 'JINYA Ramen Bar_Program'.\n"
            "9. `Reach`: The number of unique users who saw the ad, e.g., 147.\n"
            "10. `Impressions`: The total number of times the ad was displayed, e.g., 163.\n"
            "11. `Adds to cart`: The number of times the ad led to items being added to the cart, e.g., 2.\n"
            "12. `Unique link clicks`: The number of unique clicks on the ad's link, e.g., 6.\n"
            "13. `Amount Spent HYP 20`: The amount of money spent on the ad, e.g., 15.1625.\n"

            "This is the result of `print(df.head())`:\n"
            "{df_str}\n\n"
            "Follow these instructions:\n"
            "{instruction_str}\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )

        self.response_synthesis_prompt_str = (
            "Given an input question, synthesize a response from the query results.\n"
            "Query: {query_str}\n\n"
            "Pandas Instructions: \n{pandas_instructions}\n\n"
            "Pandas Output: {pandas_output}\n\n"
            "Response: "
        )

        self.pandas_prompt = PromptTemplate(self.pandas_prompt_str).partial_format(
            instruction_str=self.instruction_str, df_str=self.df.head(5)
        )

        self.pandas_output_parser = PandasInstructionParser(self.df)
        self.response_synthesis_prompt = PromptTemplate(self.response_synthesis_prompt_str)

        self.qp = QP(
            modules={
                "input": InputComponent(),
                "pandas_prompt": self.pandas_prompt,
                "text2pandas_llm": self.llm,
                "pandas_output_parser": self.pandas_output_parser,
                "response_synthesis_prompt": self.response_synthesis_prompt,
                "response_synthesis_llm": self.llm,
            },
            verbose=True,
        )
        self.qp.add_chain(["input", "pandas_prompt", "text2pandas_llm", "pandas_output_parser"])
        self.qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        self.qp.add_link("text2pandas_llm", "response_synthesis_prompt", dest_key="pandas_instructions")
        self.qp.add_link("pandas_output_parser","response_synthesis_prompt",dest_key="pandas_output")
        self.qp.add_link("response_synthesis_prompt", "response_synthesis_llm")


    def query_response(self, question):
        response, intermediates = self.qp.run_with_intermediates(
        query_str= question, )

        # print(intermediates['pandas_output_parser'].inputs['input'].message.content)
        return response.message.content

# Example usage
if __name__ == "__main__":
    csv_path = 'JINYA Sales 2022 and 2023 YTD.csv'
    df = pd.read_excel('Preprocessed_Jinya_Search_Performance_RAW_2022_Oct_2023.xlsx')
    sales_query_pipeline = SQLQueryPipeline(df)
    question = "What is the total net sales amount for JINYA Ramen Bar - Austin on April 10, 2022?"
    response = sales_query_pipeline.query_response(question)
    print(response)
