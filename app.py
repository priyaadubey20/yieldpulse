if st.button("Run AI Root Cause Analysis"):
    if not api_key:
        st.warning("Please configure your GEMINI_API_KEY to generate AI insights.")
    else:
        with st.spinner("Analyzing telemetry logs & running diagnostic models..."):
            try:
                client = genai.Client(api_key=api_key)
                
                prompt = f"""
                You are an expert AdTech Yield Analytics Lead. Analyze the following telemetry log data for publisher domain '{selected_domain}':

                {filtered_df.to_string()}

                Provide a concise, executive-level diagnostic breakdown containing:
                1. **Root Cause Analysis**: What caused the drop in RPM and rise in Loss-to-Revenue (L2R)?
                2. **Technical Diagnosis**: Identify specific issues (e.g., misfiring ad tags, low bid density, schema mismatch).
                3. **Actionable Remediation**: Provide 3 step-by-step actions for product operations and ad ops teams to resolve the issue immediately.
                """

                # Target active model endpoint directly
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )

                st.markdown(response.text)
            except Exception as e:
                # Fallback to gemini-1.5-flash if 2.5 is unavailable on the project tier
                try:
                    response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt
                    )
                    st.markdown(response.text)
                except Exception as fallback_e:
                    st.error(f"Failed to generate analysis: {fallback_e}")
